import numpy as np
import cv2
from packaging import version
from scipy.spatial.transform import Rotation as R
import socket
import json

# socket used for sending poses at the end
udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_posed(image, rvec, tvec, cam_pose, port=33051, host='169.254.158.110'):

    dst, jacobian = cv2.Rodrigues(rvec)
    pose_mat = np.vstack((np.hstack((dst,tvec)), np.array((0,0,0,1))))
    # this is a required correction due to the coordinate system convention of the PV camera
    x_180 = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    rot_temp = R.from_euler('xyz', (90,0,180), degrees=True)  # convention: x→y→z
    R_mat_temp = rot_temp.as_matrix()
    # Turn into 4x4 homogeneous (no translation)
    T_temp = np.eye(4)
    T_temp[:3, :3] = R_mat_temp
    world_pose = cam_pose @ x_180 @ pose_mat @ T_temp
    r = world_pose[0:3,0:3]
    t = world_pose[0:3,3]
    q = R.from_matrix(r)
    q = q.as_quat()
    # the negatives here are to convert to Unity's left-handed coordinate system
    position = {"x": t[0],
                "y": t[1],
                "z": -t[2],}
    rotation = {"x": -q[0],
                "y": -q[1],
                "z": q[2],
                "w": q[3]}
    output = {"pos": position,
            "rot": rotation}
    # prepare the JSON message for the pose
    msg = json.dumps(output).encode("ascii")
    udpSocket.sendto(msg, (host, port))
    return image
