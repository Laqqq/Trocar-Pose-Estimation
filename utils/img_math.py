import numpy as np
import time
from numba import njit
from scipy.spatial.transform import Rotation as R
import math


# @njit(cache=True)
def project_point(R, t, K, point) -> tuple:
    """
    Projects a 3D point onto an image using a 3x4 projection matrix.

    Parameters:
    - R: (3x3) Rotation matrix for reference frame of the world point relative to camera 
    - t: (3x1) Translation vector for reference frame of the world point relative to camera 
    - K: (3x3) Camera intrinsics matrix
    - point: length-3 or length-4 array (3D point in Cartesian or homogeneous coords)

    Returns:
    - (u, v): projected pixel coordinates as rounded ints tuple
    """
    # Convert to homogeneous coordinates if needed
    pt = np.asarray(point)
    if pt.shape[0] == 3:
        pt = np.append(pt, 1.0)
    elif pt.shape[0] != 4:
        raise ValueError("Point must be length 3 or 4")

    # P = K @ np.hstack((R, t.reshape(-1,1))) # camera projection matrix
    P = K @ np.column_stack((R, t)) # camera projection matrix
    
    # Project and normalize
    proj = P @ pt  # shape (3,)
    u = proj[0] / proj[2]
    v = proj[1] / proj[2]

    return (int(round(u)), int(round(v)))

def project_point_np(R, t, K, point) -> tuple:
    """
    Projects a 3D point onto an image using a 3x4 projection matrix.

    Parameters:
    - R: (3x3) Rotation matrix for reference frame of the world point relative to camera 
    - t: (3x1) Translation vector for reference frame of the world point relative to camera 
    - K: (3x3) Camera intrinsics matrix
    - point: length-3 or length-4 array (3D point in Cartesian or homogeneous coords)

    Returns:
    - (u, v): projected pixel coordinates as rounded ints tuple
    """
    # Convert to homogeneous coordinates if needed
    pt = np.asarray(point)
    if pt.shape[0] == 3:
        pt = np.append(pt, 1.0)
    elif pt.shape[0] != 4:
        raise ValueError("Point must be length 3 or 4")

    # P = K @ np.hstack((R, t.reshape(-1,1))) # camera projection matrix
    P = K @ np.column_stack((R, t)) # camera projection matrix
    
    # Project and normalize
    proj = P @ pt  # shape (3,)
    u = proj[0] / proj[2]
    v = proj[1] / proj[2]

    return np.array([u, v], dtype=np.float64)


# def point_to_line_distance(point, line_start, line_end) -> float:
#     """
#     Compute the perpendicular distance from a point to an infinite 2D line
#     defined by two distinct endpoints.

#     Parameters
#     ----------
#     point : (x0, y0)
#         The query point.
#     line_start : (x1, y1)
#         One endpoint of the line.
#     line_end : (x2, y2)
#         The other endpoint of the line.

#     Returns
#     -------
#     distance : float
#         The shortest (perpendicular) distance from `point` to the line through
#         `line_start` and `line_end`.
#     """
    
#     x0, y0 = point
#     x1, y1 = line_start
#     x2, y2 = line_end

#     # Direction vector of the line
#     dx = x2 - x1
#     dy = y2 - y1

#     norm = np.linalg.norm(dx, dy)

#     # unit vector
#     ux = dx / norm
#     uy = dy / norm


#     distance = abs(ux * (y0 - y1) - uy * (x0 - x1))

#     return distance



def point_to_line_distance(point, line_tuple):
    """
    Compute the perpendicular distance from a point to an 2D line
    defined by two endpoints
    """
    line_start = line_tuple[0]
    line_end = line_tuple[1]
    
    # direction vector of the line
    v = line_end - line_start
    # vector from line_start to the query point
    w = point - line_start

    # import pdb; pdb.set_trace()

    # 2D "cross product" (scalar) = v_x * w_y - v_y * w_x
    cross = v[0]*w[1] - v[1]*w[0]

    # distance = |cross| / ||v||
    return np.abs(cross) / np.linalg.norm(v)




def point_to_line_signed_distance(point, line_tuple):
    """
    Compute the signed perpendicular distance from a 2D point to a line
    defined by two endpoints. Positive if the point lies to the "left"
    of the directed line (start→end), negative if to the "right".
    """
    line_start, line_end = line_tuple

    # direction vector of the line
    v = line_end - line_start
    # vector from line_start to the query point
    w = point - line_start

    # 2D “cross product” scalar = v_x * w_y - v_y * w_x
    cross = v[0] * w[1] - v[1] * w[0]

    # signed distance = cross / ||v||
    return cross / np.linalg.norm(v)


# @njit
# def point_to_line_signed_distance_numba(px, py,
#                                      x0, y0,
#                                      x1, y1):
#     """
#     Numba-jitted signed distance from (px, py) to line through (x0,y0)->(x1,y1).
#     """
#     # direction vector
#     vx = x1 - x0
#     vy = y1 - y0
#     # vector to point
#     wx = px - x0
#     wy = py - y0

#     # 2D cross product
#     cross = vx * wy - vy * wx

#     # norm of v
#     norm_v = (vx*vx + vy*vy) ** 0.5

#     return cross / norm_v



@njit
def point_to_line_signed_distance_numba(point, line_start, line_end):
    """
    Numba-jitted signed perpendicular distance.
    `point`, `line_start`, `line_end` are all 1D float64 arrays length 2.
    """
    vx = line_end[0]   - line_start[0]
    vy = line_end[1]   - line_start[1]
    wx = point[0]      - line_start[0]
    wy = point[1]      - line_start[1]

    cross = vx * wy - vy * wx
    norm_v = math.sqrt(vx*vx + vy*vy)

    return cross / norm_v

















# Example usage:
if __name__ == "__main__":
    # Sample 3x4 projection matrix (fx=fy=1, no translation)
    P = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]], dtype=float)
    
    # A test 3D point
    point_3d = np.array([2.3, 4.7, 5.2])

    start_time = time.time()

    for i in range(10000):
        uv = project_point(P, point_3d)

    total_time = time.time() - start_time
    print("Projected coordinates:", uv)
    print(f"Time: {total_time:03f}")


def euler_to_rotm(angles, order='xyz', degrees=True):
    """
    angles : array‐like of length 3
        The Euler angles (in the convention specified by `order`).
    order : str, default 'xyz'
        Sequence of axes for rotations.  e.g. 'xyz', 'zyx', 'ZYX', etc.
    degrees : bool, default False
        Are your input angles in degrees?  If True, they'll be converted.

    Returns
    -------
    rotm : (3,3) ndarray
        The equivalent rotation matrix.
    """
    # Create a Rotation object, then extract its matrix
    rot = R.from_euler(order, angles, degrees=degrees)
    return rot.as_matrix()