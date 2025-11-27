"""SAM2-based shaft tracking with PCA-style cylinder edge extraction."""
import cv2
import numpy as np
from sam2.build_sam import build_sam2_object_tracker

MASK_ALPHA = 0.7
MASK_BETA = 0.3
MASK_COLOR = (0, 255, 0)
EDGE_COLOR = (255, 255, 0)
CENTER_COLOR = (0, 255, 255)
MIN_MASK_PIXELS = 30
MIN_POINTS_PER_BIN = 10


def axis_coordinates(pts_xy, axis_origin, axis_dir):
    """Project points onto an axis and compute signed perpendicular distances."""
    v = pts_xy - axis_origin
    s_coord = np.dot(v, axis_dir)
    d_coord = v[:, 0] * axis_dir[1] - v[:, 1] * axis_dir[0]
    return s_coord, d_coord


def to_int_point(pt):
    """Convert a floating point coordinate into integer pixel indices."""
    arr = np.asarray(pt).astype(float).ravel()
    return int(round(arr[0])), int(round(arr[1]))


def ensure_bgr(image):
    """Guarantee that the image has three channels for drawing."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def gather_mask_points(mask):
    """Collect mask coordinates as (x, y) float32 points."""
    ys, xs = np.nonzero(mask)
    if xs.size < MIN_MASK_PIXELS:
        return None
    return np.column_stack([xs, ys]).astype(np.float32)


def principal_axes(points):
    """Return the PCA center, major axis, and orthogonal axis."""
    mean, eigvecs = cv2.PCACompute(points, mean=None, maxComponents=2)
    center = mean[0]
    axis = eigvecs[0]
    axis /= np.linalg.norm(axis) + 1e-8
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    return center, axis, normal


def width_consistency_ok(s_coord, d_coord, s_min, s_max, width, robust_q, nbins, max_width_var):
    """Check whether local width variation stays within tolerance."""
    bin_edges = np.linspace(s_min, s_max, nbins + 1)
    widths = []
    for i in range(nbins):
        mask = (s_coord >= bin_edges[i]) & (s_coord < bin_edges[i + 1])
        if mask.sum() < MIN_POINTS_PER_BIN:
            continue
        d_slice = d_coord[mask]
        w_bin = np.quantile(d_slice, 1.0 - robust_q) - np.quantile(d_slice, robust_q)
        widths.append(w_bin)
    if widths and np.std(widths) > max_width_var * width:
        return False
    return True


def build_edge_set(center, axis, normal, span, offsets, include_side_edges):
    """Construct the center and optional side segments."""
    s_min, s_max = span
    left_offset, right_offset = offsets
    center_edge = (center + s_min * axis, center + s_max * axis)

    if not include_side_edges:
        return (center_edge, None, None)

    left_edge = (
        center + s_min * axis + left_offset * normal,
        center + s_max * axis + left_offset * normal,
    )
    right_edge = (
        center + s_min * axis + right_offset * normal,
        center + s_max * axis + right_offset * normal,
    )

    avg_left = left_edge[0][0] + left_edge[1][0]
    avg_right = right_edge[0][0] + right_edge[1][0]
    if avg_left > avg_right:
        left_edge, right_edge = right_edge, left_edge

    return (center_edge, left_edge, right_edge)


def draw_edge_segments(image,edges):
    """Overlay the detected edge segments."""
    disp = image.copy()
    center_edge, left_edge, right_edge = edges
    cv2.line(
        disp,
        to_int_point(center_edge[0]),
        to_int_point(center_edge[1]),
        CENTER_COLOR,
        1,
        cv2.LINE_AA,
    )
    if left_edge is not None and right_edge is not None:
        cv2.line(
            disp,
            to_int_point(left_edge[0]),
            to_int_point(left_edge[1]),
            EDGE_COLOR,
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            disp,
            to_int_point(right_edge[0]),
            to_int_point(right_edge[1]),
            EDGE_COLOR,
            1,
            cv2.LINE_AA,
        )
    return disp


def blend_mask_overlay(image, mask):
    """Apply a green overlay constrained to the mask footprint."""
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return image
    disp = image.copy()
    masked_pixels = disp[mask_bool].astype(np.float32)
    overlay = np.full(masked_pixels.shape, MASK_COLOR, dtype=np.float32)
    blended_pixels = cv2.addWeighted(masked_pixels, MASK_ALPHA, overlay, MASK_BETA, 0)
    disp[mask_bool] = blended_pixels
    return disp


def find_cylinder_edges(frame, mask_fullres, min_aspect=1.0, max_width_var=0.8, robust_q=0.05, nbins=20, camera_type="rgb"):
    """Locate the cylindrical shaft edges via PCA on the segmentation mask."""
    disp = ensure_bgr(frame)
    pts = gather_mask_points(mask_fullres)
    if pts is None:
        print("find_cylinder_edges: too few pixels in mask.")
        return disp, None

    center, axis, normal = principal_axes(pts)
    s_coord, d_coord = axis_coordinates(pts, center, axis)

    s_min, s_max = s_coord.min(), s_coord.max()
    left_q = np.quantile(d_coord, robust_q)
    right_q = np.quantile(d_coord, 1.0 - robust_q)

    length = s_max - s_min
    width = right_q - left_q
    aspect = length / (width + 1e-6)
    if aspect < min_aspect:
        print(f"find_cylinder_edges: reject aspect ratio {aspect:.2f} < {min_aspect}.")
        return disp, None

    if not width_consistency_ok(
        s_coord, d_coord, s_min, s_max, width, robust_q, nbins, max_width_var
    ):
        print("find_cylinder_edges: width variability too high.")
        return disp, None

    include_side_edges = camera_type == "rgb"
    edges = build_edge_set(center, axis, normal, (s_min, s_max), (left_q, right_q), include_side_edges)
    disp = draw_edge_segments(disp, edges)
    disp = blend_mask_overlay(disp, mask_fullres)
    return disp, edges


class SAM2PointTracker:
    """SAM2-based shaft tracking using point prompts."""

    def __init__(
        self,
        config_file,
        ckpt_path,
        camera_type,
        device="cuda:0",
        num_objects=1,
        frame_rate=30,
    ):
        self.sam = build_sam2_object_tracker(
            num_objects=num_objects,
            config_file=config_file,
            ckpt_path=ckpt_path,
            device=device,
            verbose=False,
        )
        self.camera_type = camera_type
        self.frame_rate = frame_rate
        self.needs_prompt = True
        self.bad_count = 0
        self.max_consecutive_bad = 5

    def process(
        self,
        frame,
        point_prompt=None,
    ):
        """Run one tracking step and return the visualization plus 2D line set."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask_prediction = self.track_frame(rgb, point_prompt)

        h, w = frame.shape[:2]
        mask_fullres = cv2.resize(mask_prediction, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_clean = self.refine_mask(mask_fullres)

        disp, edges = find_cylinder_edges(frame, mask_clean, camera_type=self.camera_type)

        if edges is None:
            self.handle_bad_detection()
            return disp, None

        self.bad_count = 0
        lines = self.pack_lines_for_camera(edges)
        return disp, lines

    def track_frame(self, rgb, point_prompt=None):
        """Track SAM2 objects, optionally initializing with a user prompt."""
        if self.needs_prompt:
            if point_prompt is None:
                raise ValueError("SAM2PointTracker requires point_prompt for initialization.")
            points = np.array([point_prompt], dtype=np.float64)
            sam_out = self.sam.track_new_object(img=rgb, points=points)
            self.needs_prompt = False
        else:
            sam_out = self.sam.track_all_objects(img=rgb)

        mask_tensor = sam_out["pred_masks"][0, 0]
        mask_np = (mask_tensor.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
        return mask_np

    def refine_mask(self, mask):
        """Keep the upper band of the mask to remove spurious lower blobs."""
        rows_with_mask = np.where(mask > 0)[0]
        if rows_with_mask.size == 0:
            return mask
        cutoff = int(np.quantile(rows_with_mask, 0.90))
        refined = mask.copy()
        refined[cutoff:, :] = 0
        return refined

    def handle_bad_detection(self):
        """Count consecutive failures and reset SAM2 when needed."""
        self.bad_count += 1
        if self.bad_count >= self.max_consecutive_bad:
            self.needs_prompt = True
            self.bad_count = 0
            self.sam.reset_tracker_state()

    def pack_lines_for_camera(self, edges):
        """Return the tuple of 2D segments expected by downstream consumers."""
        center_edge, left_edge, right_edge = edges
        if self.camera_type == "rgb":
            if left_edge is None or right_edge is None:
                raise RuntimeError("RGB camera requires left and right edge segments.")
            return (center_edge, right_edge, left_edge)
        return (center_edge,)
