"""Dataset utilities and preprocessing helpers for trocar pose estimation."""

import json
import os
import pickle
import sys

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

HL2SS_VIEWER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'hl2ss', 'viewer')
)
if HL2SS_VIEWER_PATH not in sys.path:
    sys.path.append(HL2SS_VIEWER_PATH)

from Rendering.VispyRenderer import VispyRenderer
from utils import PVCalibrationToOpenCVFormat

PNG_EXTENSION = '.png'
EXPECTED_GT_2D_POINTS = 11
CALIBRATION_FILENAME = '1280_720_calibration.pkl'
PLY_MODEL_PATH = 'trocar_fixed_short_ascii.ply'
MAX_BBOX_EXPANSION = 20
MAX_EXPANSION_ATTEMPTS = 10
REAL_TRAIN_RATIO = 0.9
REAL_FRACTION = 0.7
SYNTH_FRACTION = 0.3
HOSPITAL_SYNTH_FRACTION = 0.7
SPLIT_SEED = 42
DATALOADER_WORKERS = 4
DEFAULT_IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
DEFAULT_IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def list_png_files(directory):
    """Return all PNG filenames within a directory, sorted for determinism."""
    return sorted(f for f in os.listdir(directory) if f.endswith(PNG_EXTENSION))


def compute_image_stats(image_dir,files,desc):
    """Compute per-channel mean and std for the provided image set."""
    means= []
    stds= []
    for filename in tqdm(files, desc=desc):
        img_path = os.path.join(image_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image for stats: {img_path}")
        normalized = image.astype(np.float32) / 255.0
        means.append(np.mean(normalized, axis=(0, 1)))
        stds.append(np.std(normalized, axis=(0, 1)))
    dataset_mean = np.mean(means, axis=0).tolist()
    dataset_std = np.mean(stds, axis=0).tolist()
    return dataset_mean, dataset_std

def load_keypoints(ann_path):
    """Load 2D keypoints stored either as JSON (default) or whitespace-delimited text."""
    with open(ann_path, 'r', encoding='utf-8') as file:
        keypoints_json = json.load(file)['keypoints']
    return np.array([[kp['x'], kp['y']] for kp in keypoints_json], dtype=np.float32)

def load_pose(pose_path):
    """Load rotation matrix and translation vector from JSON or text pose files."""
    with open(pose_path, 'r', encoding='utf-8') as file:
        pose_data = json.load(file)
    rotation = np.array(pose_data['rotation_matrix'])
    translation = np.array(pose_data['translation']).reshape(-1, 1)
    return rotation, translation

class KeypointDataset(Dataset):
    """Torch dataset that serves cropped trocar images, keypoints, and optional masks."""

    def __init__(self, preprocessed_dir, use_mask=False, transform=None):
        self.image_dir = os.path.join(preprocessed_dir, 'images')
        self.ann_dir = os.path.join(preprocessed_dir, 'annotations')
        self.mask_dir = os.path.join(preprocessed_dir, 'masks') if use_mask else None
        self.files = list_png_files(self.image_dir)
        self.use_mask = use_mask
        self.transform = transform
        self.keypoints_3d = np.array(
            [
                [-0.00001400, 0.00706800, -0.12432900],  # 00 tip_apex
                [-0.00104700, -0.01985400, 0.02522100],  # 01 base_apex
                [0.00058362, 0.01584000, 0.02472500],    # 02 ringB_x+
                [0.01348600, 0.00093375, 0.02223200],    # 03 ringB_y+
                [-0.00043952, -0.01201900, 0.01972300],  # 04 ringB_x-
                [-0.01363200, 0.00197730, 0.02221600],   # 05 ringB_y-
                [0.00312768, 0.02507700, -0.00989168],   # 06 ringA_x+
                [0.01298600, 0.00773140, -0.01511800],   # 07 ringA_y+
                [-0.00034296, -0.00949815, -0.01956606],  # 08 ringA_x-
                [-0.01301400, 0.00889688, -0.01491216],   # 09 ringA_y-
                [0.00072326, 0.00973550, -0.01867657],    # 10 ringA_center
                [0.00041618, 0.00626321, 0.02222400],     # 11 ringB_center
            ],
            dtype=np.float32,
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = self.files[idx]
        image = self.read_image(img_file)
        gt_2d_points = self.read_keypoints(img_file)
        mask = self.read_mask(img_file) if self.use_mask else None
        image, mask, gt_2d_points = self.apply_transforms(image, mask, gt_2d_points)
        keypoints_tensor = torch.from_numpy(gt_2d_points).float()
        if self.use_mask:
            return image, mask, keypoints_tensor
        return image, keypoints_tensor

    def read_image(self, filename):
        img_path = os.path.join(self.image_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        return image

    def read_keypoints(self, filename):
        ann_path = os.path.join(self.ann_dir, filename.replace(PNG_EXTENSION, '.json'))
        return load_keypoints(ann_path)

    def read_mask(self, filename):
        if not self.mask_dir:
            raise ValueError("Mask directory was not provided but masks were requested.")
        mask_path = os.path.join(self.mask_dir, filename)
        seg_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if seg_mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        return seg_mask.astype(np.float32) / 255.0

    def apply_transforms(self,image,mask,keypoints):
        """Apply albumentations transforms or fall back to default normalization."""
        if not self.transform:
            return self.default_transform(image, mask, keypoints)

        keypoints_list = keypoints.tolist()
        transformed = (
            self.transform(image=image, keypoints=keypoints_list, mask=mask)
            if self.use_mask and mask is not None
            else self.transform(image=image, keypoints=keypoints_list)
        )
        transformed_image = transformed['image']
        transformed_keypoints = np.array(transformed['keypoints'], dtype=np.float32)
        transformed_mask = None
        if self.use_mask and 'mask' in transformed:
            transformed_mask = transformed['mask']
            if transformed_mask.ndim == 2:
                transformed_mask = transformed_mask.unsqueeze(0)
            transformed_mask = transformed_mask.float()
        return transformed_image, transformed_mask, transformed_keypoints

    def default_transform(image,mask,keypoints):
        """Basic normalization when no albumentations pipeline is provided."""
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image_tensor = (image_tensor - DEFAULT_IMAGE_MEAN) / DEFAULT_IMAGE_STD
        return image_tensor, mask, keypoints

def compute_dataset_stats(preprocessed_dir):
    """Compute RGB mean and std over every preprocessed image."""
    image_dir = os.path.join(preprocessed_dir, 'images')
    files = list_png_files(image_dir)
    if not files:
        raise ValueError("No images found in preprocessed_dir for stats computation.")
    dataset_mean, dataset_std = compute_image_stats(
        image_dir, files, desc="Computing dataset stats"
    )
    print(f"Dataset mean: {dataset_mean}")
    print(f"Dataset std: {dataset_std}")
    return dataset_mean, dataset_std

def compute_real_dataset_stats(preprocessed_dir, real_files):
    """Compute RGB mean and std using only real (non-synthetic) images."""
    if not real_files:
        raise ValueError("No real images found for stats computation.")
    image_dir = os.path.join(preprocessed_dir, 'images')
    dataset_mean, dataset_std = compute_image_stats(
        image_dir, real_files, desc="Computing real dataset stats"
    )
    print(f"Real dataset mean: {dataset_mean}")
    print(f"Real dataset std: {dataset_std}")
    return dataset_mean, dataset_std

def preprocess_data(
    image_dir,
    ann_dir,
    pose_dir,
    preprocessed_dir,
    is_json_ann=True,
    is_json_pose=True,
    prefix='',
    crop_size=256,
    use_mask=False,
    mask_dir=None,
):
    """Crop, pad, and normalize annotations for downstream training."""
    with open(CALIBRATION_FILENAME, 'rb') as file:
        calibration_data = pickle.load(file)
    intrinsics_opencv, _ = PVCalibrationToOpenCVFormat(calibration_data)
    renderer = None
    if mask_dir is None:
        renderer = VispyRenderer(
            width=1280,
            height=720,
            camera_matrix=intrinsics_opencv,
            ply_model_path=PLY_MODEL_PATH,
        )

    os.makedirs(os.path.join(preprocessed_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_dir, 'annotations'), exist_ok=True)
    if use_mask:
        os.makedirs(os.path.join(preprocessed_dir, 'masks'), exist_ok=True)

    all_files = list_png_files(image_dir)
    valid_count = 0
    print(f"Preprocessing images from {image_dir} using CAD model projections...")
    for filename in tqdm(all_files, desc="Preprocessing images"):
        img_path = os.path.join(image_dir, filename)
        ann_ext = '.json' if is_json_ann else '.txt'
        pose_ext = '.json' if is_json_pose else '.txt'
        ann_path = os.path.join(ann_dir, filename.replace(PNG_EXTENSION, ann_ext))
        pose_path = os.path.join(pose_dir, filename.replace(PNG_EXTENSION, pose_ext))

        image = cv2.imread(img_path)
        if image is None:
            print(f"image is none: {img_path}")
            continue
        height, width = image.shape[:2]

        if not os.path.exists(ann_path):
            print(f"annotation is none: {ann_path}")
            continue
        gt_2d_points = load_keypoints(ann_path)
        if gt_2d_points.shape[0] != EXPECTED_GT_2D_POINTS:
            print(f"keypoint is none or invalid count for {ann_path}")
            continue

        if mask_dir is None:
            if not os.path.exists(pose_path):
                print(f"pose is none: {pose_path}")
                continue
            trocar_R, trocar_t = load_pose(pose_path)
            if renderer is None:
                raise RuntimeError("Renderer was not initialized for mask generation.")
            mask = renderer.render_mask(trocar_R, trocar_t)
        else:
            mask_path = os.path.join(mask_dir, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"mask is none for {mask_path}")
                continue
            mask = mask > 0

        if not np.any(mask):
            print(f"mask is none or empty for {filename}")
            continue

        rows, cols = np.nonzero(mask)
        y1 = max(0, np.min(rows))
        y2 = min(height, np.max(rows) + 1)
        x1 = max(0, np.min(cols))
        x2 = min(width, np.max(cols) + 1)

        original_x1, original_y1, original_x2, original_y2 = x1, y1, x2, y2
        for _ in range(MAX_EXPANSION_ATTEMPTS):
            expand_left = np.random.randint(0, MAX_BBOX_EXPANSION + 1)
            expand_right = np.random.randint(0, MAX_BBOX_EXPANSION + 1)
            expand_top = np.random.randint(0, MAX_BBOX_EXPANSION + 1)
            expand_bottom = np.random.randint(0, MAX_BBOX_EXPANSION + 1)
            new_x1 = max(0, original_x1 - expand_left)
            new_y1 = max(0, original_y1 - expand_top)
            new_x2 = min(width, original_x2 + expand_right)
            new_y2 = min(height, original_y2 + expand_bottom)
            if (
                np.all(gt_2d_points[:, 0] >= new_x1)
                and np.all(gt_2d_points[:, 0] <= new_x2)
                and np.all(gt_2d_points[:, 1] >= new_y1)
                and np.all(gt_2d_points[:, 1] <= new_y2)
            ):
                x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
                break
        else:
            continue

        crop_img = image[y1:y2, x1:x2]
        crop_h, crop_w = crop_img.shape[:2]
        if crop_h == 0 or crop_w == 0:
            continue

        adjusted_gt_2d_points = gt_2d_points - np.array([x1, y1])
        crop_mask = None
        if use_mask:
            crop_mask = (mask[y1:y2, x1:x2] > 0).astype(np.float32)

        scale = crop_size / max(crop_h, crop_w)
        new_h, new_w = int(crop_h * scale), int(crop_w * scale)
        crop_img_resized = cv2.resize(crop_img, (new_w, new_h))
        crop_mask_resized = None
        if use_mask and crop_mask is not None:
            crop_mask_resized = cv2.resize(
                crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )

        pad_h_top = (crop_size - new_h) // 2
        pad_h_bottom = crop_size - new_h - pad_h_top
        pad_w_left = (crop_size - new_w) // 2
        pad_w_right = crop_size - new_w - pad_w_left
        padded_img = np.pad(
            crop_img_resized,
            ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right), (0, 0)),
            mode='constant',
        )

        padded_mask = None
        if use_mask and crop_mask_resized is not None:
            padded_mask = np.pad(
                crop_mask_resized,
                ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)),
                mode='constant',
            )

        adjusted_gt_2d_points *= scale
        adjusted_gt_2d_points += np.array([pad_w_left, pad_h_top])

        saved_img_file = prefix + filename
        cv2.imwrite(os.path.join(preprocessed_dir, 'images', saved_img_file), padded_img)

        adjusted_keypoints_json = {
            'keypoints': [{'x': float(x), 'y': float(y)} for x, y in adjusted_gt_2d_points]
        }
        with open(
            os.path.join(preprocessed_dir, 'annotations', saved_img_file.replace(PNG_EXTENSION, '.json')),
            'w',
            encoding='utf-8',
        ) as ann_out:
            json.dump(adjusted_keypoints_json, ann_out)

        if use_mask and padded_mask is not None:
            padded_mask_uint8 = (padded_mask * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(preprocessed_dir, 'masks', saved_img_file),
                padded_mask_uint8,
            )
        valid_count += 1
    print(f"Preprocessed and saved {valid_count} valid images out of {len(all_files)} from {image_dir}")

def preprocess_all(base_dir,preprocessed_dir,use_mask=False,crop_size=256):
    """Batch preprocess helper (currently covers the 'new' dataset splits)."""
    for sub in range(1, 14):
        sub_dir = str(sub)
        image_dir = os.path.join(base_dir, sub_dir, 'output', 'rgb')
        ann_dir = os.path.join(base_dir, sub_dir, 'output', 'keypoints_prop')
        pose_dir = os.path.join(base_dir, sub_dir, 'output', 'pose')
        if not os.path.exists(image_dir):
            continue
        prefix = f'new{sub_dir}_'
        preprocess_data(
            image_dir,
            ann_dir,
            pose_dir,
            preprocessed_dir,
            prefix=prefix,
            use_mask=use_mask,
            crop_size=crop_size,
            mask_dir=None,
        )


def get_dataloaders(preprocessed_dir,batch_size=16,use_mask=False):
    """Create train/validation dataloaders with the prescribed real/synthetic ratios."""

    def sample_with_oversubscription(pool_indices,desired,label):
        if not pool_indices or desired <= 0:
            return []
        pool_size = len(pool_indices)
        if desired <= pool_size:
            selection = torch.randperm(pool_size)[:desired].tolist()
            return [pool_indices[i] for i in selection]
        print(f"oversample {label}")
        num_copies = (desired // pool_size) + 1
        permuted = torch.randperm(pool_size).tolist() * num_copies
        selection = permuted[:desired]
        return [pool_indices[i] for i in selection]

    images_dir = os.path.join(preprocessed_dir, 'images')
    existing_images = list_png_files(images_dir)
    if existing_images:
        print("Preprocessed data found, skipping preprocessing.")
    else:
        raise ValueError("No preprocessed data found or empty. Run preprocess_all first.")

    full_dataset = KeypointDataset(preprocessed_dir, use_mask=use_mask, transform=None)
    files = full_dataset.files
    file_to_idx = {filename: idx for idx, filename in enumerate(files)}

    # Split files into real and synthetic
    real_files = [f for f in files if f.startswith(('old_', 'new'))]
    # Split synthetic files into hospital and other
    synth_hosp_files = [f for f in files if f.startswith('synth_output_hospital')]
    synth_other_files = [
        f for f in files if f.startswith('synth_') and not f.startswith('synth_output_hospital')
    ]

    real_idxs = [file_to_idx[f] for f in real_files]
    synth_hosp_idxs = [file_to_idx[f] for f in synth_hosp_files]
    synth_other_idxs = [file_to_idx[f] for f in synth_other_files]

    num_real = len(real_idxs)
    num_hosp = len(synth_hosp_idxs)
    num_other = len(synth_other_idxs)
    print(f"Real: {num_real}, Synth Hospital: {num_hosp}, Synth Other: {num_other}")

    dataset_mean, dataset_std = compute_real_dataset_stats(preprocessed_dir, real_files)
    train_transform = A.Compose(
        [
            A.GaussianBlur(blur_limit=(1, 3)),
            A.ISONoise(),
            A.GaussNoise(),
            A.CLAHE(),
            A.CoarseDropout(max_height=16, max_width=16, min_width=8, min_height=8),
            A.ColorJitter(hue=0.1),
            A.Normalize(mean=dataset_mean, std=dataset_std),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    )
    val_transform = A.Compose(
        [
            A.Normalize(mean=dataset_mean, std=dataset_std),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    )

    val_txt = os.path.join(preprocessed_dir, 'val_files.txt')
    torch.manual_seed(SPLIT_SEED)
    if os.path.exists(val_txt):
        with open(val_txt, 'r', encoding='utf-8') as vf:
            val_files = [line.strip() for line in vf if line.strip()]
        val_files = [f for f in val_files if f in real_files]
        val_ids = [file_to_idx[f] for f in val_files]
        train_real_idxs = [idx for idx in real_idxs if idx not in val_ids]
        print(
            f"Loaded existing split (filtered to real): {len(train_real_idxs)} train_real, {len(val_ids)} val"
        )
    else:
        idxs = torch.randperm(num_real)
        train_size = int(REAL_TRAIN_RATIO * num_real)
        train_subids = idxs[:train_size].tolist()
        val_subids = idxs[train_size:].tolist()
        train_real_idxs = [real_idxs[j] for j in train_subids]
        val_ids = [real_idxs[j] for j in val_subids]
        val_files = [real_files[j] for j in val_subids]
        with open(val_txt, 'w', encoding='utf-8') as vf:
            for file in sorted(val_files):
                vf.write(f"{file}\n")
        print(
            f"Created and saved new split on real: {len(train_real_idxs)} train_real, {len(val_ids)} val"
        )

    num_real_train = len(train_real_idxs)
    if num_real_train == 0:
        raise ValueError("No real training data.")

    # Calculate desired number of synthetic data for training
    desired_synth = int(num_real_train * SYNTH_FRACTION / REAL_FRACTION)
    desired_hosp = int(desired_synth * HOSPITAL_SYNTH_FRACTION)
    desired_other = desired_synth - desired_hosp
    # Sample synthetic data for training
    selected_hosp_idxs = sample_with_oversubscription(
        synth_hosp_idxs, desired_hosp, label='hospital'
    )
    selected_other_idxs = sample_with_oversubscription(
        synth_other_idxs, desired_other, label='other'
    )

    train_ids = train_real_idxs + selected_hosp_idxs + selected_other_idxs
    print(
        "Train set: %d samples (real: %d, synth_hosp: %d, synth_other: %d)"
        % (len(train_ids), num_real_train, len(selected_hosp_idxs), len(selected_other_idxs))
    )

    train_dataset = KeypointDataset(preprocessed_dir, use_mask=use_mask, transform=train_transform)
    val_dataset = KeypointDataset(preprocessed_dir, use_mask=use_mask, transform=val_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=SubsetRandomSampler(train_ids),
        num_workers=DATALOADER_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=SubsetRandomSampler(val_ids),
        num_workers=DATALOADER_WORKERS,
    )
    return train_loader, val_loader