"""Training entrypoint for keypoint detection with self-generated masks."""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from data_preparation import get_dataloaders
from keypointNet import KeypointDetectionNet, generate_gt_heatmaps, combined_heatmap_coord_loss

BEST_MODEL_PATH = ''
CHECKPOINT_DIR = ''
SYN_IMAGE_DIR = ''

def to_uint8_bgr_from_imagenet(tensor):
    """Convert a normalized tensor into an RGB uint8 image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = tensor.detach().cpu().float()
    image = image * std + mean
    image = (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return image.permute(1, 2, 0).contiguous().numpy()

def tensor_to_rgb_u8(img, mean=None, std=None):
    """Convert a tensor or array to RGB uint8, optionally de-normalizing."""
    if isinstance(img, torch.Tensor):
        data = img.detach().cpu().float()
        if data.dim() == 3 and data.shape[0] in (1, 3):
            data = data if data.shape[0] == 3 else data.repeat(3, 1, 1)
            data = data.permute(1, 2, 0)
        elif data.dim() == 2:
            data = data.unsqueeze(-1).repeat(1, 1, 3)
        arr = data.numpy()
    else:
        arr = img

    arr = np.asarray(arr, dtype=np.float32)
    if mean is not None and std is not None:
        mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        arr = arr * std_arr + mean_arr
    if arr.max() <= 1.0 + 1e-6:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def probmap_to_uint8_gray(prob_map):
    """Convert a probability map into a grayscale uint8 image."""
    if isinstance(prob_map, torch.Tensor):
        pm = prob_map.detach().cpu().float()
        if pm.dim() == 3 and pm.size(0) == 1:
            pm = pm[0]
        elif pm.dim() == 3 and pm.size(0) != 1:
            pm = pm[0]
        arr = pm.numpy()
    else:
        arr = np.asarray(prob_map, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0, None)
    vmax = arr.max()
    if vmax <= 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.rint(arr / vmax * 255.0).astype(np.uint8)

def probmap_to_color_rgb(prob_map, colormap=cv2.COLORMAP_JET):
    """Map a probability array to a colored RGB heatmap."""
    gray = probmap_to_uint8_gray(prob_map)
    bgr = cv2.applyColorMap(gray, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def overlay_heatmap_on_rgb(rgb_img, prob_map, alpha=0.5, colormap=cv2.COLORMAP_JET, mean=None, std=None):
    """Blend a probability heatmap on top of an RGB image."""
    rgb = tensor_to_rgb_u8(rgb_img, mean=mean, std=std)
    heat_rgb = probmap_to_color_rgb(prob_map)
    if heat_rgb.shape[:2] != rgb.shape[:2]:
        heat_rgb = cv2.resize(heat_rgb, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(rgb, 1.0 - alpha, heat_rgb, alpha, 0.0)

def lerp(current_epoch, total_epochs, start_tau=1.0, end_tau=0.25):
    """Linearly interpolate tau over the configured epoch range."""
    clamped_epoch = max(0, min(current_epoch, total_epochs))
    progress = float(clamped_epoch) / float(max(total_epochs, 1))
    delta = end_tau - start_tau
    tau = start_tau + progress * delta
    return tau

def prepare_batch(batch, use_mask, device):
    """Split dataloader output and move tensors to the target device."""
    if use_mask:
        img, mask, gt_points = batch
        mask = mask.to(device)
    else:
        img, gt_points = batch
        mask = None
    return img.to(device), gt_points.to(device), mask


def train(model, train_loader, val_loader, epochs, lr, use_heatmap=True, use_mask=True, device='cuda'):
    """Main training loop orchestrating optimization and validation."""
    _ = use_heatmap  # Flag retained for compatibility
    model.to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-5)
    best_val_mpjpe = float('inf')
    vis_dir = os.path.join(os.path.dirname(SYN_IMAGE_DIR.rstrip('/')), 'visualized')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        tau_start = 0.77
        tau_min = 0.65
        tau_stop_epoch = 300

        if epoch < tau_stop_epoch:
            tau = lerp(epoch, tau_stop_epoch, tau_start, tau_min)
        else:
            tau = tau_min
        # tau = 0.65
        sigma = 5.0
        print(f'tau: {tau}')
        print(f'sigma: {sigma}')
        for batch in tqdm(train_loader):
            img, gt_points, mask = prepare_batch(batch, use_mask, device)
            logits, pred_mask_logit = model(img)
            gt_heatmaps = generate_gt_heatmaps(gt_points, img.shape[2], img.shape[3], sigma, device, normalize="sum")
            total_loss, heatmap_loss, coord_loss, mean_px_err, pred_points, prob_map = combined_heatmap_coord_loss(
                logits, gt_points, gt_heatmaps, tau=tau, lambda_coord=1.0, use_kl_for_heatmap=True
            )
            seg_loss = 10 * F.binary_cross_entropy_with_logits(pred_mask_logit, mask, reduction='mean')
            total_loss = total_loss + seg_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
        train_loss /= len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')
        print(f'Heatmap loss: {heatmap_loss}')
        print(f'Coord loss: {coord_loss}')
        print(f'Segmentation loss: {seg_loss}')
        # Validation
        model.eval()
        val_loss = 0
        val_pck = 0
        val_mpjpe = 0
        num_batches = len(val_loader)

        first_batch = True
        with torch.no_grad():
            for batch in val_loader:
                img, gt_points, mask = prepare_batch(batch, use_mask, device)
                logits, pred_mask_logit = model(img)
                gt_heatmaps = generate_gt_heatmaps(gt_points, img.shape[2], img.shape[3], sigma, device, normalize="sum")
                total_loss, heatmap_loss, coord_loss, mean_px_err, pred_points, prob_map = combined_heatmap_coord_loss(
                    logits, gt_points, gt_heatmaps, tau=tau, lambda_coord=1.0
                )
                seg_loss = 20 * F.binary_cross_entropy_with_logits(pred_mask_logit, mask, reduction='mean')
                total_loss = total_loss + seg_loss
                # Optional: Visualize ground-truth mask if available
                if first_batch:
                    heatmap_dir = os.path.join(vis_dir, 'heatmaps', f'{epoch:05d}')
                    gt_dir = os.path.join(vis_dir, 'gt', f'{epoch:05d}')
                    gt_heatmap_dir = os.path.join(vis_dir, 'gt_heatmaps', f'{epoch:05d}')
                    seg_mask_dir = os.path.join(vis_dir, 'seg_mask', f'{epoch:05d}')
                    os.makedirs(gt_heatmap_dir, exist_ok=True)
                    os.makedirs(heatmap_dir, exist_ok=True)
                    os.makedirs(gt_dir, exist_ok=True)
                    os.makedirs(seg_mask_dir, exist_ok=True)
                    pred_prob = torch.sigmoid(pred_mask_logit[0]).detach().cpu()
                    gt_img = to_uint8_bgr_from_imagenet(img[0])
                    img_seg_overlay = overlay_heatmap_on_rgb(
                        gt_img.copy(), 
                        pred_prob, 
                        alpha=0.5,
                        colormap=cv2.COLORMAP_JET
                    )
                    # Save as PNG (convert RGB to BGR for cv2.imwrite)
                    pred_seg_filename = os.path.join(seg_mask_dir, 'predicted_mask.png')
                    cv2.imwrite(pred_seg_filename, cv2.cvtColor(img_seg_overlay, cv2.COLOR_RGB2BGR))
                    print(f"Saved predicted segmask overlay to: {pred_seg_filename}")
                    if use_mask and mask is not None and torch.any(mask[0] > 0):
                        gt_mask = mask[0].detach().cpu().float()
                        img_gt_seg_overlay = overlay_heatmap_on_rgb(
                            gt_img.copy(), 
                            gt_mask, 
                            alpha=0.5,
                            colormap=cv2.COLORMAP_JET
                        )
                        gt_seg_filename = os.path.join(seg_mask_dir, 'gt_mask.png')
                        cv2.imwrite(gt_seg_filename, cv2.cvtColor(img_gt_seg_overlay, cv2.COLOR_RGB2BGR))
                        print(f"Saved GT segmask overlay to: {gt_seg_filename}")

                    for i in range(prob_map.shape[1]):
                        point_xy = gt_points[0, i]
                        point_xy = point_xy.detach()
                        point_xy = point_xy.cpu()
                        point_xy = point_xy.round()
                        point_xy = point_xy.to(torch.int32)
                        xy_list  = point_xy.tolist()
                        xy_tuple = tuple(xy_list)   
                        img_plus_heatmap = overlay_heatmap_on_rgb(gt_img.copy(), probmap_to_color_rgb(prob_map[0][i]), alpha=0.5)
                        img_plus_gt_heatmap = overlay_heatmap_on_rgb(gt_img.copy(), probmap_to_color_rgb(gt_heatmaps[0][i]), alpha=0.5)

                        cv2.circle(img_plus_heatmap, xy_tuple, 1, (0, 255, 0), -1)
                        heatmap_filename = os.path.join(heatmap_dir, f'{i:02d}.png')
                        cv2.imwrite(heatmap_filename, img_plus_heatmap)                        

                        cv2.circle(gt_img, xy_tuple, 1, (0, 255, 0), -1)
                        gt_filename = os.path.join(gt_dir, f'{i:02d}.png')
                        cv2.imwrite(gt_filename, gt_img)

                        gt_heatmap_filename = os.path.join(gt_heatmap_dir, f'{i:02d}.png')
                        cv2.imwrite(gt_heatmap_filename, img_plus_gt_heatmap)

                    first_batch = False


                val_loss += total_loss.item()
                mpjpe = torch.mean(torch.norm(pred_points - gt_points, dim=2)).item()
                val_mpjpe += mpjpe
                diag = 0.05 * torch.sqrt(torch.tensor(img.shape[2] ** 2 + img.shape[3] ** 2)).item()
                dist = torch.norm(pred_points - gt_points, dim=2)
                pck = (dist < diag).float().mean() * 100
                val_pck += pck.item()
        val_loss /= num_batches
        val_mpjpe /= num_batches
        val_pck /= num_batches
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val MPJPE: {val_mpjpe:.4f} pixels') # Add this
        print(f'Val PCKh@0.05: {val_pck:.2f}%') # Renamed to PCKh for clarity
        # Save best model
        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f'Saved best model with Val Loss: {best_val_mpjpe:.4f}')

        current_epoch_checkpoint_filename = os.path.join(CHECKPOINT_DIR, f'{epoch}.pth')
        torch.save(model.state_dict(), current_epoch_checkpoint_filename)
        print('saved to: ', current_epoch_checkpoint_filename)

        last_learn_rate = scheduler.get_last_lr()
        print("last_learn_rate", last_learn_rate)
        if (last_learn_rate[0]) > 0.000001:
            scheduler.step()
        else:
            print("min learn rate reached")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='', help='Path to dataset directory')
    parser.add_argument('--dataset_type', type=str, default='', help='Type of dataset that need to be preprocess')
    parser.add_argument('--num_subset', type=int, default=-1)
    parser.add_argument('--preprocessed_dir', type=str, default='/home/arssist/trocar_tracking/trocar_processing/dataset/preprocess_comb_ext_newmask', help='Path to preprocessed output directory')
    # parser.add_argument('--yolo_path', type=str, required=True, help='Path to trained YOLO model (if needed for fallback cropping)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_heatmap', action='store_true', help='Use heatmap loss instead of direct MSE')
    parser.add_argument('--use_mask', action='store_true', help='Use segmentation mask as input')
    # parser.add_argument('--sigma', type=float, default=5.0, help='Sigma for Gaussian heatmaps if using heatmap')
    args = parser.parse_args()
    
    if args.base_dir and args.dataset_type and args.num_subset > 0:
        preprocess_all(args.base_dir, args.preprocessed_dir, args.dataset_type, args.num_subset,use_mask=args.use_mask)

    # #########################
    train_loader, val_loader = get_dataloaders(args.preprocessed_dir, batch_size=args.batch_size, use_mask=args.use_mask)
    model = KeypointDetectionNet(num_keypoints=11)
    train(model, train_loader, val_loader, args.epochs, args.lr, args.use_heatmap, args.use_mask)