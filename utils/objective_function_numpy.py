import numpy as np
from img_math import *
from utils.drawing_utils import *

def flatten_residuals(res_list):
    """
    Take a list whose elements are either scalars or array‐likes,
    and return one flat 1D numpy array of floats.
    """
    out = []
    for r in res_list:
        a = np.asanyarray(r)
        if a.ndim == 0:
            out.append(float(a))          # scalar
        else:
            out.extend(a.ravel().tolist())  # multi-element residual
    return np.array(out, dtype=float)

# --- 6D rotation <-> matrix conversions (Zhou et al. CVPR 2019) ---
def matrix_to_rot6d(R):
    """Extract first two columns of 3×3 R into a length-6 vector."""
    a6 = R.ravel(order='F')[0:6]
    return a6


def rot6d_to_matrix(a6):
    """Reconstruct a valid 3×3 rotation from 6D rep via Gram–Schmidt + cross-product."""
    M = a6.reshape(2,3).T
    # Gram–Schmidt orthonormalization
    b1 = M[:, 0]
    b1 = b1 / np.linalg.norm(b1)
    # remove component of a2 along b1
    proj = np.dot(b1, M[:, 1])
    b2 = M[:, 1] - proj * b1
    b2 = b2 / np.linalg.norm(b2)
    # b3 is orthogonal cross-product
    b3 = np.cross(b1, b2)
    # stack into rotation matrix
    R = np.column_stack((b1, b2, b3))
    return R

# main objective function
def compute_error(params,
                  R0, t0,
                  K, extrinsic,
                  u_target,
                  w_t, w_r, w_p,
                  point3d):
    """
    params: length-9 consiting of [a6d, t] 
    R0: initial rotation matrix (3x3)
    t0:   initial translation (3,)
    K:    intrinsic (3×3)
    extrinsic: camera-to-world 4×4 (note, you may need to invert beforehand) 
    u_target: image point (2,)
    w_t, w_r, w_p: weights
    point3d: 3D point in object coords (homogenous) (4,)
    """
    # translation error
    t = params[6:9]
    err_t = np.linalg.norm(t0-t)
    # rotation error via Frobenius norm squared
    a6 = params[0:6]
    R = rot6d_to_matrix(a6)
    Delta = R0.T @ R
    err_r = 6 - 2 * np.trace(Delta)
    # build full pose 4×4
    pose = np.zeros((4,4), dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = t
    pose[3, 3] = 1
    # can speed this up probably
    M = extrinsic @ pose
    point2d = K @ M[0:3, :] @ point3d
    point2d = point2d / point2d[2]
    delta_t = point2d[0:2] - u_target
    err_p = np.dot(delta_t, delta_t)
    return w_t * err_t + w_r * err_r + w_p * err_p


def make_obj_func(R0, t0,
                  K, extrinsic,
                  u_target,
                  point3d,
                  w_t=1.0, w_r=1.0, w_p=1.0):

    def obj(params):
        return compute_error(params,
                             R0, t0,
                             K, extrinsic,
                             u_target,
                             w_t, w_r, w_p,
                             point3d)
    return obj

# main objective function
def compute_error_line(params,
                        R0, t0,
                        K,
                        img_line_1,
                        w_t, w_r, w_l, w_p, w_3d, w_rxry, w_tz,
                        point3d,
                        point2d_0):
    """
    params: length-9 consiting of [a6d, t] 
    R0: initial rotation matrix (3x3)
    t0:   initial translation (3,)
    K:    intrinsic (3×3)
    img_line_pt_1: first image point on target line (2,)
    img_line_pt_2: second image point on target line (2,)
    w_t, w_r, w_p: weights
    point3d: Array of 3D points in object coords which should lie on the line (homogenous and as columns) (4, N)
    point2d: Array of 2D points in image coords of the original 3D points when projected onto the image (N, 2)
    """
    # err_t: translation error
    t = params[6:9]
    err_t = np.linalg.norm(t0-t)

    dtz = t[2]-t0[2]
    err_tz = dtz*dtz

    a6 = params[0:6]
    R = rot6d_to_matrix(a6)
    delta_R = R0.T @ R
    err_r = 6 - 2 * np.trace(delta_R)

    rot_vec, _ = cv2.Rodrigues(delta_R)      # returns a 3×1 vector
    rx, ry, rz = rot_vec.ravel() 
    err_rxry = rx*rx + ry*ry

    pose_0 = np.zeros((4,4), dtype=np.float64)
    pose_0[:3, :3] = R0
    pose_0[:3, 3] = t0
    pose_0[3, 3] = 1
    points_3d_in_cam_space_0 = pose_0 @ point3d

    pose = np.zeros((4,4), dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = t
    pose[3, 3] = 1

    points_3d_in_cam_space = pose @ point3d
    points2d = K @ points_3d_in_cam_space[0:3, :]

    err_3D = np.sum(np.linalg.norm(points_3d_in_cam_space_0-points_3d_in_cam_space, axis=0))
    err_l = 0
    err_p = 0

    for i in range(points2d.shape[1]):
        point2d = points2d[:,i] / points2d[2,i]
        p2d = point2d[0:2]
        err_l += point_to_line_distance(p2d, img_line_1)
        err_p += np.linalg.norm(p2d - point2d_0[i])
    return w_t    * err_t    + \
           w_r    * err_r    + \
           w_l    * err_l    + \
           w_p    * err_p    + \
           w_3d   * err_3D   + \
           w_rxry * err_rxry + \
           w_tz   * err_tz


def make_obj_func_line(R0, t0,
                        K,
                        img_line_1,
                        point3d,
                        point2d,
                        w_t=1.0,
                        w_r=1.0,
                        w_l=1.0,
                        w_p=1.0,
                        w_3d=1.0,
                        w_rxry=1.0,
                        w_tz=1.0,
                        ):

    def obj(params):
        return compute_error_line(params,
                                R0, t0,
                                K,
                                img_line_1,
                                w_t, w_r, w_l, w_p, w_3d, w_rxry, w_tz,
                                point3d,
                                point2d)
    return obj

def compute_error_multi_line(params,
                                R0, t0,
                                K,
                                img_lines,
                                w_t, w_r, w_l, w_p, w_3d, w_rxry, w_tz, w_square, w_gl,
                                point3d,
                                point2d_0,
                                obj_pts_square,
                                img_pts_square_0,
                                rgb_extrinsics,
                                left_camera_calibration,
                                right_camera_calibration,
                                left_shaft_line,
                                right_shaft_line):
    """
    params: length-9 consiting of [a6d, t] 
    R0: initial rotation matrix (3x3)
    t0:   initial translation (3,)
    K:    intrinsic (3×3)
    img_line_pt_1: first image point on target line (2,)
    img_line_pt_2: second image point on target line (2,)
    w_t, w_r, w_p: weights
    point3d: Array of 3D points in object coords which should lie on the line (homogenous and as columns) (4, N)
    point2d: Array of 2D points in image coords of the original 3D points when projected onto the image (N, 2)
    """
    # err_t: translation error
    t = params[6:9]
    err_t = np.linalg.norm(t0-t)
    dtz = t[2]-t0[2]
    err_tz = dtz*dtz
    # err_r: rotation error via Frobenius norm squared
    a6 = params[0:6]
    R = rot6d_to_matrix(a6)
    delta_R = R0.T @ R
    err_r = 6 - 2 * np.trace(delta_R)
    # this term basically only allows roation about the camera frame's z axis.
    # I think this should be done better by only allowing a rotation about
    # the axis between the pose origin and the camera origin, but the
    # camera z axis is a close enough optimization.
    rot_vec, _ = cv2.Rodrigues(delta_R)      # returns a 3×1 vector
    rx, ry, rz = rot_vec.ravel() 
    err_rxry = rx*rx + ry*ry
    # err_l: point to line error
    # build full pose 4×4
    pose_0 = np.zeros((4,4), dtype=np.float64)
    pose_0[:3, :3] = R0
    pose_0[:3, 3] = t0
    pose_0[3, 3] = 1
    points_3d_in_cam_space_0 = pose_0 @ point3d
    pose = np.zeros((4,4), dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = t
    pose[3, 3] = 1
    # can speed this up probably
    points_3d_in_cam_space = pose @ point3d
    points2d = K @ points_3d_in_cam_space[0:3, :]
    err_3D = np.sum(np.linalg.norm(points_3d_in_cam_space_0-points_3d_in_cam_space, axis=0))
    err_l = 0
    err_p = 0
    # if using a lot of points, this can be sped up.
    for i in range(points2d.shape[1]):
        single_point2d = points2d[:,i] / points2d[2,i]
        p2d = single_point2d[0:2]
        # err_l += point_to_line_distance(p2d, img_lines[i//2])
        err_l += point_to_line_distance(p2d, img_lines)
        err_p += np.linalg.norm(p2d - point2d_0[i])
    err_square = 0
    square_points_3d_in_cam_space = pose @ obj_pts_square
    square_points2d = K @ square_points_3d_in_cam_space[0:3, :]
    for i in range(square_points2d.shape[1]):
        square_single_point2d = square_points2d[:,i] / square_points2d[2,i]
        square_p2d = square_single_point2d[0:2]
        err_square += np.linalg.norm(square_p2d - img_pts_square_0[i])
    shaft_pts = point3d[:,:2]
    K_left_4x4, pose_left_4x4 = left_camera_calibration
    K_right_4x4, pose_right_4x4 = right_camera_calibration
    pts_in_left_frame = pose_left_4x4 @ np.linalg.inv(rgb_extrinsics) @ pose @ shaft_pts
    pts_in_right_frame = pose_right_4x4 @ np.linalg.inv(rgb_extrinsics) @ pose @ shaft_pts
    img_pts_left = K_left_4x4 @ pts_in_left_frame
    img_pts_left /= img_pts_left[2,:]
    img_pts_right = K_right_4x4 @ pts_in_right_frame
    img_pts_right /= img_pts_right[2,:]
    err_gray_lines = 0
    for i in range(len(img_pts_left)):
        err_gray_lines += point_to_line_distance(img_pts_left[i], left_shaft_line)
        err_gray_lines += point_to_line_distance(img_pts_right[i], right_shaft_line)
    return w_t    * err_t    + \
           w_r    * err_r    + \
           w_l    * err_l    + \
           w_p    * err_p    + \
           w_3d   * err_3D   + \
           w_rxry * err_rxry + \
           w_tz   * err_tz   + \
           w_square * err_square + \
           w_gl * err_gray_lines


def make_obj_func_multi_line(R0, t0,
                                K,
                                img_lines,
                                point3d,
                                point2d,
                                obj_pts_square,
                                img_pts_square,
                                rgb_extrinsics,
                                left_camera_calibration,
                                right_camera_calibration,
                                left_shaft_line,
                                right_shaft_line,
                                w_t=1.0,
                                w_r=1.0,
                                w_l=1.0,
                                w_p=1.0,
                                w_3d=1.0,
                                w_rxry=1.0,
                                w_tz=1.0,
                                w_square=1.0,
                                w_gl=1.0,
                                ):

    def obj(params):
        # import pdb; pdb.set_trace()
        return compute_error_multi_line(params,
                                R0, t0,
                                K,
                                img_lines,
                                w_t, w_r, w_l, w_p, w_3d, w_rxry, w_tz, w_square, w_gl,
                                point3d,
                                point2d,
                                obj_pts_square,
                                img_pts_square,
                                rgb_extrinsics,
                                left_camera_calibration,
                                right_camera_calibration,
                                left_shaft_line,
                                right_shaft_line)
    return obj





def pose_residuals(r,
                    R0, t0, pose0,
                    K,
                    points_3d_in_cam_space_0,
                    img_lines,
                    point3d,
                    point2d_0,
                    obj_pts_pnp,
                    img_pts_pnp_0,
                    rgb_extrinsics,
                    left_camera_calibration,
                    right_camera_calibration,
                    left_shaft_line,
                    right_shaft_line,
                    w_l,
                    w_gl,
                    w_pnp):

    res = []
    R, _ = cv2.Rodrigues(r)
    pose = np.zeros((4,4), dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = t0
    pose[3, 3] = 1
    points_3d_in_cam_space = pose @ point3d
    points2d = K @ points_3d_in_cam_space[0:3, :]
    rgb_shaft_points_img = (points2d / points2d[2, :])[:2,:]
    for i in range(rgb_shaft_points_img.shape[1]):
        line_pt1, line_pt2 = img_lines
        res.append( w_l * point_to_line_signed_distance_numba((rgb_shaft_points_img[:,i]).flatten(), line_pt1, line_pt2))
    pnp_points_3d_in_cam_space = pose @ obj_pts_pnp
    pnp_points2d = K @ pnp_points_3d_in_cam_space[0:3, :]
    pnp_points_img = (pnp_points2d / pnp_points2d[2, :])[:2,:]

    res.append(w_pnp * (pnp_points_img - img_pts_pnp_0.T))

 
    # grayscale shaft line
    shaft_pts = point3d[:,:2]

    K_left_4x4, pose_left_4x4 = left_camera_calibration
    K_right_4x4, pose_right_4x4 = right_camera_calibration

    pts_in_left_frame = pose_left_4x4 @ np.linalg.inv(rgb_extrinsics) @ pose @ shaft_pts
    pts_in_right_frame = pose_right_4x4 @ np.linalg.inv(rgb_extrinsics) @ pose @ shaft_pts

    img_pts_left = K_left_4x4 @ pts_in_left_frame
    img_pts_left /= img_pts_left[2,:]

    img_pts_right = K_right_4x4 @ pts_in_right_frame
    img_pts_right /= img_pts_right[2,:]

    for i in range(img_pts_left.shape[1]):
        left_line_pt1, left_line_pt2 = left_shaft_line
        res.append(w_gl * point_to_line_signed_distance_numba((img_pts_left[:2,i]).flatten(), left_line_pt1, left_line_pt2))
        right_line_pt1, right_line_pt2 = right_shaft_line
        res.append(w_gl * point_to_line_signed_distance_numba((img_pts_right[:2,i]).flatten(), right_line_pt1, right_line_pt2))
    return flatten_residuals(res)



def pose_residuals_with_t(x,
                            R0, t0, pose0,
                            K,
                            points_3d_in_cam_space_0,
                            img_lines,
                            point3d,
                            point2d_0,
                            obj_pts_pnp,
                            img_pts_pnp_0,
                            rgb_extrinsics,
                            left_camera_calibration,
                            right_camera_calibration,
                            left_shaft_line,
                            right_shaft_line,
                            w_t,
                            w_l,
                            w_gl,
                            w_pnp):

    res = []
    r = x[0:3]
    t = x[3:6]
    R, _ = cv2.Rodrigues(r)
    pose = np.zeros((4,4), dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = t
    pose[3, 3] = 1
    res.append(w_t * (t - t0))

    points_3d_in_cam_space = pose @ point3d
    points2d = K @ points_3d_in_cam_space[0:3, :]
    rgb_shaft_points_img = (points2d / points2d[2, :])[:2,:]
    for i in range(rgb_shaft_points_img.shape[1]):
        line_pt1, line_pt2 = img_lines[i//2]
        res.append( w_l * point_to_line_signed_distance_numba((rgb_shaft_points_img[:,i]).flatten(), line_pt1, line_pt2))
    pnp_points_3d_in_cam_space = pose @ obj_pts_pnp
    pnp_points2d = K @ pnp_points_3d_in_cam_space[0:3, :]
    pnp_points_img = (pnp_points2d / pnp_points2d[2, :])[:2,:]

    res.append(w_pnp * (pnp_points_img - img_pts_pnp_0.T))
    shaft_pts = point3d[:,:2]
    K_left_4x4, pose_left_4x4 = left_camera_calibration
    K_right_4x4, pose_right_4x4 = right_camera_calibration
    pts_in_left_frame = pose_left_4x4 @ np.linalg.inv(rgb_extrinsics) @ pose @ shaft_pts
    pts_in_right_frame = pose_right_4x4 @ np.linalg.inv(rgb_extrinsics) @ pose @ shaft_pts

    img_pts_left = K_left_4x4 @ pts_in_left_frame
    img_pts_left /= img_pts_left[2,:]
    img_pts_right = K_right_4x4 @ pts_in_right_frame
    img_pts_right /= img_pts_right[2,:]

    for i in range(img_pts_left.shape[1]):
        left_line_pt1, left_line_pt2 = left_shaft_line
        res.append(w_gl * point_to_line_signed_distance_numba((img_pts_left[:2,i]).flatten(), left_line_pt1, left_line_pt2))
        right_line_pt1, right_line_pt2 = right_shaft_line
        res.append(w_gl * point_to_line_signed_distance_numba((img_pts_right[:2,i]).flatten(), right_line_pt1, right_line_pt2))
    return flatten_residuals(res)