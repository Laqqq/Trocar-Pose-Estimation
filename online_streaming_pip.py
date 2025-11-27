"""Real-time streaming pipeline for trocar pose estimation on HoloLens footage."""

import argparse
import os
import sys
import threading
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hl2ss', 'viewer')))

import cv2
import hl2ss
import hl2ss_imshow
import hl2ss_lnm
import numpy as np
import torch
from pynput import keyboard
from ultralytics import YOLO

from Rendering.VispyRenderer import VispyRenderer
from utils.data_gui_window import DataWindow
from utils.drawing_utils import draw_pose_coord_frame_in_image
from keypointNet import KeypointDetectionNet
from utils.markerless import send_posed
from utils.nn_shaft_detection import SAM2PointTracker
from utils.optim import pose_optimization_least_squares_with_t
from utils.utils import PVCalibrationToOpenCVFormat

cv2.namedWindow('PV overlay (CAD)', cv2.WINDOW_NORMAL)
cv2.namedWindow('LEFT overlay', cv2.WINDOW_NORMAL)
cv2.namedWindow('RIGHT overlay', cv2.WINDOW_NORMAL)

# HoloLens address
host = "169.254.158.110"

# Enable Mixed Reality Capture (Holograms)
enable_mrc = False

# Enable Shared Capture
# If another program is already using the PV camera, you can still stream it by
# enabling shared mode, however you cannot change the resolution and framerate
shared = False

# Camera parameters
# Ignored in shared mode
width     = 1280
height    = 720
framerate = 30

# MODE_1 means that the camera pose is streamed in addition to the image
mode = hl2ss.StreamMode.MODE_1

# Video encoding profile and bitrate (None = default)
profile = hl2ss.VideoProfile.H265_MAIN
bitrate = None  # should figure out the bitrate later...

decoded_format = 'bgr24' # opencv format

# set fixed focal length for the PV camera
pv_focus = 600 # in mm

YOLO_WTS = ''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = KeypointDetectionNet(num_keypoints=11).to(device).eval()
model.load_state_dict(torch.load('', map_location=device))
try:
    yolo = YOLO(YOLO_WTS)
except Exception as e:
    print(f"[WARN] YOLO load failed: {e}")
    yolo = None

MEAN = torch.tensor([0.4197433590888977, 0.4314931333065033, 0.4453215003013611], device=device).view(3,1,1)
STD = torch.tensor([0.28419291973114014, 0.2886916995048523, 0.2951403856277466], device=device).view(3,1,1)

dist_coeffs = np.zeros((5,1), dtype=np.float64)

SAVE_IMAGE = True  # already present in your code; leave as-is
SAVE_EVERY_N = 1   # save every N frames (1 = every frame)
SAVE_ROOT = os.path.join(os.path.dirname(__file__), 'saved_frames')

if SAVE_IMAGE:
    run_tag = time.strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(SAVE_ROOT, run_tag)
    pv_overlay_dir = os.path.join(run_dir, 'pv_overlay')
    os.makedirs(pv_overlay_dir, exist_ok=True)

def preprocess_crop(img, box, out_size=256):
    """Crop, resize, and pad an ROI while tracking the transform parameters."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    x1 = x1
    y1 = y1
    x2 = x2
    y2 = y2
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    scale = out_size / max(ch, cw) if max(ch, cw) > 0 else 1.0
    new_h, new_w = max(1, int(ch * scale)), max(1, int(cw * scale))
    crop_resized = cv2.resize(crop, (new_w, new_h))
    pad_top = (out_size - new_h) // 2
    pad_bottom = out_size - new_h - pad_top
    pad_left = (out_size - new_w) // 2
    pad_right = out_size - new_w - pad_left
    padded = np.pad(crop_resized, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    return padded, (scale, pad_left, pad_top, x1, y1)


def detect_box_yolo(img):
    """Run YOLO inference and return the largest detected bounding box."""
    if yolo is None:
        return None
    res = yolo.predict(img, conf=0.2, iou=0.5, verbose=False)
    if not res or res[0].boxes is None or len(res[0].boxes) == 0:
        return None
    boxes = res[0].boxes.xyxy.cpu().numpy().astype(int) # (N,4)
    # pick the largest box
    areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    idx = int(np.argmax(areas))
    return boxes[idx]

#------------------------------------------------------------------------------
dataWindow = DataWindow() # create GUI window for viewing stream info

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

def HL2_receive_thread(client, lock, shared_data, stop_event):
    """Continuously fetch PV packets and store the latest result."""
    while not stop_event.is_set():
        data = client.get_next_packet()
        with lock:
            shared_data['latest_data'] = data

def spatial_softmax(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Compute a numerically stable spatial softmax for each heatmap."""
    B, K, H, W = logits.shape
    tau = max(tau, 1e-8)
    x = logits / tau
    x = x - x.amax(dim=(2, 3), keepdim=True)
    prob = torch.softmax(x.flatten(2), dim=-1).view(B, K, H, W)
    return prob
def soft_argmax_2d(logits: torch.Tensor, tau: float = 1.0):
    """Return the expected (x, y) coordinates from heatmap logits."""
    B, K, H, W = logits.shape
    prob = spatial_softmax(logits, tau=tau)
    xs = torch.arange(W, device=logits.device, dtype=logits.dtype).view(1,1,1,W)
    ys = torch.arange(H, device=logits.device, dtype=logits.dtype).view(1,1,H,1)
    mu_x = (prob * xs).sum(dim=(2,3))
    mu_y = (prob * ys).sum(dim=(2,3))
    return torch.stack([mu_x, mu_y], dim=-1)
def apply_mask_overlay(image, mask, color=(255, 0, 0), alpha=0.5):
    """Blend a binary mask on top of an RGB image."""
    overlay = image.copy()
    m = np.where(mask)
    c = np.array(color, dtype=np.float32)
    for ch in range(3):
        overlay[m[0], m[1], ch] = (1 - alpha) * image[m[0], m[1], ch] + alpha * c[ch]
    return overlay.astype(np.uint8)


def main():
    """Stream PV frames, run detection, solve pose, and visualize overlays."""
    keypoints_3d = np.array([
        [-0.00104700, -0.01985400, 0.02522100], # 01 base_apex
        [ 0.00058362, 0.01584000, 0.02472500], # 02 ringB_x+
        [ 0.01348600, 0.00093375, 0.02223200], # 03 ringB_y+
        [-0.00043952, -0.01201900, 0.01972300], # 04 ringB_x-
        [-0.01363200, 0.00197730, 0.02221600], # 05 ringB_y-
        [ 0.00312768, 0.02507700, -0.00989168], # 06 ringA_x+
        [ 0.01298600, 0.00773140, -0.01511800], # 07 ringA_y+
        [-0.00034296, -0.00949815, -0.01956606], # 08 ringA_x-
        [-0.01301400, 0.00889688, -0.01491216], # 09 ringA_y-
        [ 0.00072326, 0.00973550, -0.01867657], # 10 ringA_center
        [ 0.00041618, 0.00626321, 0.02222400], # 11 ringB_center
    ], dtype=np.float32)

    frame_idx = 0

    IMG_SIZE = 256
    TAU = 0.722
    ren_cache = {}

    PLY_PATH = 'trocar_fixed_ascii.ply'

    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, None, enable_mrc=enable_mrc, shared=shared)

    ipc_rc = hl2ss_lnm.ipc_rc(host, hl2ss.IPCPort.REMOTE_CONFIGURATION, None)
    ipc_rc.open()
    ipc_rc.pv_wait_for_subsystem(True)
    ipc_rc.pv_set_focus(hl2ss.PV_FocusMode.Manual, hl2ss.PV_AutoFocusRange.Normal, hl2ss.PV_ManualFocusDistance.Infinity, pv_focus, hl2ss.PV_DriverFallback.Disable)
    ipc_rc.close()

    # First fetch the factory calibration
    calibrationData = hl2ss_lnm.download_calibration_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, None, width, height, framerate)
    # printHl2ssCalibration(calibrationData)

    # convert the calibration data to opencv conventions.
    intrinsics_opencv, extrinsics_opencv = PVCalibrationToOpenCVFormat(calibrationData)

    client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, profile=profile, bitrate=bitrate, decoded_format=decoded_format)

    shared_data = {
        'latest_data': None,
    }

    stop_event = threading.Event()

    lock = threading.Lock()

    def on_press(key):
        nonlocal stop_event
        if (key == keyboard.Key.esc):
            # Signal the threads to stop
            stop_event.set()

    listener = keyboard.Listener(on_press=on_press)

    receive_thread = threading.Thread(target=HL2_receive_thread, args=(client, lock, shared_data, stop_event))
        
    listener.start()
    client.open()
    receive_thread.start()

    while not stop_event.is_set():
        with lock:
            data = shared_data['latest_data']

        if (data is not None):
            dataWindow.update_data(data)  # Send data to the GUI

            cam_pose_opencv = data.pose.T

            if data.payload.image is not None:
                cv2.imshow('Video_1', data.payload.image)
                frame_idx += 1
                h, w = data.payload.image.shape[:2]
                # Renderer for PV (cache per (w,h))
                if 'pv' not in ren_cache or ren_cache['pv']['size'] != (w,h):
                    ren_cache['pv'] = {
                        'ren': VispyRenderer(width=w, height=h, camera_matrix=intrinsics_opencv, ply_model_path=PLY_PATH),
                        'size': (w,h)
                        }
                ren = ren_cache['pv']['ren']
                box = detect_box_yolo(data.payload.image)
                if box is None:
                    print("Yolo Fail")
                    continue

                box[0] = box[0] - 20
                box[2] = box[2] + 20
                box[1] = box[1] - 20
                box[3] = box[3] + 20
                if box is None:
                    box = (0, 0, w, h)
                padded, (scale, pad_left, pad_top, x1, y1) = preprocess_crop(data.payload.image, box, out_size=IMG_SIZE)
                inp = torch.from_numpy(padded.transpose(2,0,1)).float().to(device) / 255.0
                inp = (inp - MEAN) / STD
                inp = inp.unsqueeze(0)
                start_model = time.time()
                with torch.no_grad():
                    logits, _mask_logit = model(inp)
                    pred_xy = soft_argmax_2d(logits, tau=TAU)
                    pred_xy = pred_xy[0].cpu().numpy()
                stop_model = time.time()
                print("model inference time is: ", stop_model - start_model)
                pred_xy -= np.array([pad_left, pad_top], dtype=np.float32)
                pred_xy /= max(scale, 1e-8)
                pred_xy += np.array([x1, y1], dtype=np.float32)
                # Solve PnP with RANSAC, then refine
                start_pnp = time.time()
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    keypoints_3d, pred_xy, intrinsics_opencv, dist_coeffs,
                    iterationsCount=600, reprojectionError=3.0, confidence=0.995,
                    flags=cv2.SOLVEPNP_EPNP
                    )
                if not ok:
                    # fallSAVE_IMAGEback: iterative without RANSAC
                    ok, rvec, tvec = cv2.solvePnP(
                        keypoints_3d, pred_xy, intrinsics_opencv, dist_coeffs,
                        flags=cv2.SOLVEPNP_EPNP
                        )
                inliers = None
                num_inliers = 0 if inliers is None else int(len(inliers))
                # refine on inliers if possible
                if ok and inliers is not None and len(inliers) >= 3:
                    idx = inliers.flatten()
                    ok_ref, rvec, tvec = cv2.solvePnP(
                        keypoints_3d[idx], pred_xy[idx], intrinsics_opencv, dist_coeffs,
                        rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP
                        )
                    ok = ok_ref
                if not ok:
                    print(f"[FAIL] PnP")
                
                stop_pnp = time.time()
                print('pnp processing time is: ', stop_pnp-start_pnp)
                reproj, _ = cv2.projectPoints(keypoints_3d, rvec, tvec, intrinsics_opencv, dist_coeffs)
                reproj = reproj.reshape(-1,2)
                rpe_all = float(np.mean(np.linalg.norm(reproj - pred_xy, axis=1)))
                R, _ = cv2.Rodrigues(rvec)
                T_pv_obj = np.eye(4, dtype=np.float64)
                T_pv_obj[:3,:3] = R
                T_pv_obj[:3, 3] = tvec.reshape(3)
                # overlay on PV
                pv_mask = ren.render_mask(R, tvec.reshape(3))
                pv_overlay = apply_mask_overlay(data.payload.image, pv_mask > 0, color=(0,0,255), alpha=0.5)
                # draw keypoints & axes on PV for sanity
                vis = pv_overlay.copy()
                for (x,y) in pred_xy.astype(int):
                    cv2.circle(vis, (x,y), 3, (0,0,255), -1)
                try:
                    cv2.drawFrameAxes(vis, intrinsics_opencv, dist_coeffs, rvec, tvec, length=0.05, thickness=2)
                except Exception:
                    pass

                R_cv, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
                det_raw = np.linalg.det(R_cv)
                if (not np.isfinite(R_cv).all()) or (np.linalg.norm(R_cv) < 1e-7) or (det_raw <= 0):
                    print("stupid non-positive determinant")
                    continue

                if SAVE_IMAGE and vis is not None and (frame_idx % SAVE_EVERY_N == 0):
                    ov_path = os.path.join(pv_overlay_dir, f'pv_overlay_{frame_idx:06d}.png')
                    cv2.imwrite(ov_path, vis)

                final_result = send_posed(data.payload.image, rvec, tvec, cam_pose_opencv)
                cv2.imshow('Video',vis)
            else:
                cv2.imshow('Video', data.payload.image)
            cv2.waitKey(1)

    client.close()
    listener.join()
    receive_thread.join()

    hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--loop',
        action='store_true',
        help="Will keep trying to connect to HL2 if the connection is reset"
    )
    
    args = parser.parse_args()

    while True:
        try:
            main()
        except ConnectionResetError:
            print("Connection reset. Trying to connect again...")

        if not args.loop:
            break