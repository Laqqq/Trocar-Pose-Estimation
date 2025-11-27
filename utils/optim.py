import numpy as np
from scipy.optimize import least_squares
import cv2
from img_math import *
from utils.drawing_utils import *
from objective_function_numpy import *
import time

def pose_optimization_least_squares_with_t(rvec,
                                    tvec,
                                    K,
                                    target_img_lines,
                                    point3d,
                                    obj_pts_square,
                                    img_pts_square,
                                    test_img,
                                    rgb_extrinsics,
                                    left_camera_calibration,
                                    right_camera_calibration,
                                    left_shaft_line,
                                    right_shaft_line):



    R0, _ = cv2.Rodrigues(rvec)
    t0 = tvec.ravel().astype(np.float64)
    pts2d = []
    for i in range(point3d.shape[1]):
        pt = point3d[:,i]
        pts2d.append(np.array(project_point(R0, t0, K, pt)))
    point2d = np.stack(pts2d)
    
    obj_pts_square_homogenous = np.vstack([obj_pts_square.T, np.ones(obj_pts_square.shape[0])])

    #Weight to tune
    w_t = 1.0
    w_l = 1500.0
    w_gl = 960.0
    w_pnp = 600.0

    r0, _ = cv2.Rodrigues(R0)
    pose_0 = np.zeros((4,4), dtype=np.float64)
    pose_0[:3, :3] = R0
    pose_0[:3, 3] = t0
    pose_0[3, 3] = 1
    points_3d_in_cam_space_0 = pose_0 @ point3d
    x0 = np.concatenate((r0.flatten(), t0))
    optimize_start_time = time.time()
    res = least_squares(
            pose_residuals_with_t, x0,
            method='trf',
            max_nfev = 1000000,
            verbose=2,
            xtol=1e-12,
            ftol=1e-12,
            args=(
                R0, t0, pose_0,
                K,
                points_3d_in_cam_space_0,
                target_img_lines,
                point3d,
                point2d, obj_pts_square_homogenous,
                img_pts_square,
                rgb_extrinsics,
                left_camera_calibration,
                right_camera_calibration,
                left_shaft_line,
                right_shaft_line,
                w_t,
                w_l,
                w_gl,
                w_pnp)
            )
    optimize_end_time = time.time()
    print("\nOptimization result:")
    print(res.x)
    print("Optimize time: ", optimize_end_time-optimize_start_time)
    r1 = res.x[0:3]
    R1, _ = cv2.Rodrigues(r1)
    t1 = res.x[3:6]
    test_img = draw_pose_coord_frame_in_image(R0, t0, K, test_img, brightness=80, scale=0.03)
    for i in range(0, point3d.shape[1], 2):
        point2d_1 = np.array(project_point(R0, t0, K, point3d[:,i]))
        point2d_2 = np.array(project_point(R0, t0, K, point3d[:,i+1]))
        if i < 2:
            color = (0, 0, 255)
        elif i < 4:
            color = (0, 0, 255)
            test_img = draw_ray_from_two_points((point2d_1, point2d_2), test_img, color)
        else:
            color = (0, 0, 255)
            test_img = draw_ray_from_two_points((point2d_1, point2d_2), test_img, color)
    result_img = test_img.copy()
    result_img = draw_pose_coord_frame_in_image(R1, t1, K, result_img, scale=0.03)
    obj_2d_ref, _ = cv2.projectPoints(obj_pts_square, R1, t1, K, distCoeffs=np.zeros(4))
    for i in range(len(obj_2d_ref)):
        original_pt = tuple(int(x) for x in img_pts_square[i].round())
        cv2.circle(result_img,original_pt,1,(0, 0, 255),-1)
        point_2d = tuple(obj_2d_ref[i].ravel().astype(int))
        cv2.circle(result_img,point_2d,1,(255, 0, 0),-1)
    for i in range(0, point3d.shape[1], 2):
        point2d_1 = np.array(project_point(R1, t1, K, point3d[:,i]))
        point2d_2 = np.array(project_point(R1, t1, K, point3d[:,i+1]))
        if i < 2:
            color = (0, 255, 255)
        elif i < 4:
            color = (0, 255, 255)
            result_img = draw_ray_from_two_points((point2d_1, point2d_2), result_img, color)
        else:
            color = (0, 255, 255)
            result_img = draw_ray_from_two_points((point2d_1, point2d_2), result_img, color)
    rvec_corrected,_ = cv2.Rodrigues(R1)
    tvec_corrected = t1

    return result_img, rvec_corrected, tvec_corrected