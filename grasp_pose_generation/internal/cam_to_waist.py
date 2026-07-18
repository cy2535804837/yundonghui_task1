from typing import Dict, Union

import numpy as np
from scipy.spatial.transform import Rotation as R


def _pose7_to_matrix(pose: Union[list, np.ndarray], mm: bool) -> np.ndarray:
    pose = np.array(pose, dtype=float)
    if pose.shape != (7,):
        raise ValueError("pose must be length-7 [x,y,z,qx,qy,qz,qw]")
    t = pose[:3] / 1000.0 if mm else pose[:3]
    q = pose[3:]
    T = np.eye(4, dtype=float)
    T[:3, 3] = t
    Rot = np.eye(4, dtype=float)
    Rot[:3, :3] = R.from_quat(q).as_matrix()
    return T @ Rot


def _matrix_to_pose_dict(T: np.ndarray) -> Dict[str, float]:
    if T.shape != (4, 4):
        raise ValueError("T must be 4x4")
    position = T[:3, 3]
    roll, pitch, yaw = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=False)
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
    }


def head_pose_to_waist(
    pose_head_m: Union[list, np.ndarray],
    T_waist_head: np.ndarray,
) -> Dict[str, float]:
    if T_waist_head.shape != (4, 4):
        raise ValueError("T_waist_head must be 4x4")
    T_tag_head = _pose7_to_matrix(pose_head_m, mm=False)
    T_tag_waist = T_waist_head @ T_tag_head
    return _matrix_to_pose_dict(T_tag_waist)

