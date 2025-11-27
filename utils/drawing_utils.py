import numpy as np
import cv2

def draw_pose_coord_frame_in_image(R, t, K, image, scale=0.05, brightness=255):
    image = image.copy()
    distortion_params = (0,0,0,0)

    rvec, _ = cv2.Rodrigues(R)

    frame_points = np.array([[0,0,0.0],[1,0,0],[0,1,0],[0,0,1]]) * scale

    points_2d, _ = cv2.projectPoints(frame_points, rvec, t, K, distortion_params)

    # draw coordinate frame
    ox = int(round(points_2d[0][0][0]))
    oy = int(round(points_2d[0][0][1]))
    
    cx = int(round(points_2d[1][0][0]))
    cy = int(round(points_2d[1][0][1]))
    cv2.line(image, (ox,oy), (cx,cy), (0, 0, brightness), 2)

    cx = int(round(points_2d[2][0][0]))
    cy = int(round(points_2d[2][0][1]))
    cv2.line(image, (ox,oy), (cx,cy), (0, brightness, 0), 2)

    cx = int(round(points_2d[3][0][0]))
    cy = int(round(points_2d[3][0][1]))
    cv2.line(image, (ox,oy), (cx,cy), (brightness, brightness, 0), 2)

    cv2.circle(image, (ox,oy), 3, (brightness, brightness, brightness), -1)

    return image


def draw_line_from_two_points(line, image, color=(0, 0, 255)):
    image = image.copy()

    pt1 = line[0]
    pt2 = line[1]

    arbitrarily_large_scalar = 100
    d = pt2 - pt1

    endpoint_1 = pt1 + d * arbitrarily_large_scalar
    endpoint_2 = pt1 - d * arbitrarily_large_scalar

    cv2.line(image,
                np.rint(endpoint_1).astype(np.int64).tolist(),
                np.rint(endpoint_2).astype(np.int64).tolist(),
                color,
                1)

    return image


def draw_ray_from_two_points(line, image, color=(0, 0, 255)):
    image = image.copy()

    pt1 = line[0]
    pt2 = line[1]

    arbitrarily_large_scalar = 100
    d = pt2 - pt1

    endpoint_1 = pt1 + d * arbitrarily_large_scalar
    endpoint_2 = pt1 

    cv2.line(image,
                np.rint(endpoint_1).astype(np.int64).tolist(),
                np.rint(endpoint_2).astype(np.int64).tolist(),
                color,
                2, lineType=cv2.LINE_AA)

    return image