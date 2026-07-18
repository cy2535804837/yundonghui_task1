#!/usr/bin/env python3
"""Transforms between SLAM map/world, mobile base, and waist_yaw_link."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from .robot_pose import get_robot_pose

DEFAULT_WAIST_FRAME = "waist_yaw_link"
DEFAULT_BASE_FRAME = "base_footprint"


def pose2d_to_matrix(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    """Planar pose ``x,y,yaw`` -> 4x4 transform ``T_world_base``."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_euler("z", float(yaw)).as_matrix()
    T[:3, 3] = [float(x), float(y), float(z)]
    return T


def trans_quat_to_matrix(translation: Sequence[float], quat_xyzw: Sequence[float]) -> np.ndarray:
    """Translation + quaternion[x,y,z,w] -> 4x4 transform."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_quat([float(v) for v in quat_xyzw]).as_matrix()
    T[:3, 3] = [float(v) for v in translation]
    return T


def pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
    """``[x,y,z]`` or ``[x,y,z,qx,qy,qz,qw]`` -> 4x4 transform."""
    import numpy as np

    vals = [float(v) for v in pose]
    if len(vals) == 3:
        T = np.eye(4, dtype=float)
        T[:3, 3] = vals
        return T
    if len(vals) == 7:
        return trans_quat_to_matrix(vals[:3], vals[3:7])
    raise ValueError(f"pose must have 3 or 7 elements, got {len(vals)}")


def matrix_to_pose7(T: np.ndarray) -> List[float]:
    """4x4 transform -> ``[x,y,z,qx,qy,qz,qw]``."""
    from scipy.spatial.transform import Rotation as R

    quat = R.from_matrix(T[:3, :3]).as_quat()
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), *[float(v) for v in quat]]


def waist_from_base_tf(
    node: Any,
    waist_frame: str = DEFAULT_WAIST_FRAME,
    base_frame: str = DEFAULT_BASE_FRAME,
    timeout: float = 3.0,
) -> np.ndarray:
    """Read ``T_waist_base`` from live TF."""
    try:
        from xarm_sdk import lookup_tf_once
    except ImportError:
        from xarm_sdk.tools import lookup_tf_once

    res = lookup_tf_once(
        node,
        target_frame=str(waist_frame),
        source_frame=str(base_frame),
        timeout=float(timeout),
    )
    if res is None:
        raise RuntimeError(f"TF lookup timed out: {waist_frame} <- {base_frame}")
    translation, quat_xyzw = res
    return trans_quat_to_matrix(translation, quat_xyzw)


def approximate_waist_from_base(waist_z_in_map: float = 0.0) -> np.ndarray:
    """Fallback ``T_waist_base`` when waist and base are treated as coincident."""
    import numpy as np

    T = np.eye(4, dtype=float)
    T[2, 3] = -float(waist_z_in_map)
    return T


def object_world_to_waist(
    object_world_pose: Sequence[float],
    robot_base_world: Dict[str, Any],
    waist_from_base: np.ndarray,
    *,
    target_frame_id: Optional[str] = None,
) -> List[float]:
    """Transform object pose from map/world frame into ``waist_yaw_link``."""
    r_frame = str(robot_base_world.get("frame_id", "map"))
    if target_frame_id is not None and str(target_frame_id) != r_frame:
        raise ValueError(
            f"target frame_id {target_frame_id!r} does not match robot pose frame {r_frame!r}"
        )
    T_world_base = pose2d_to_matrix(
        float(robot_base_world["x"]),
        float(robot_base_world["y"]),
        float(robot_base_world["yaw"]),
    )
    T_obj_world = pose_to_matrix(object_world_pose)
    import numpy as np

    T_obj_waist = waist_from_base @ np.linalg.inv(T_world_base) @ T_obj_world
    return matrix_to_pose7(T_obj_waist)


def object_world_to_waist_approx(
    object_world_pose: Sequence[float],
    robot_base_world: Dict[str, Any],
    *,
    target_frame_id: Optional[str] = None,
    waist_z_in_map: float = 0.0,
) -> List[float]:
    """Approximate transform with waist/base coincident in X/Y/Yaw."""
    r_frame = str(robot_base_world.get("frame_id", "map"))
    if target_frame_id is not None and str(target_frame_id) != r_frame:
        raise ValueError(
            f"target frame_id {target_frame_id!r} does not match robot pose frame {r_frame!r}"
        )
    vals = [float(v) for v in object_world_pose]
    if len(vals) not in (3, 7):
        raise ValueError(f"pose must have 3 or 7 elements, got {len(vals)}")
    base = planar_map_delta_to_base(
        vals[0],
        vals[1],
        float(robot_base_world["x"]),
        float(robot_base_world["y"]),
        float(robot_base_world["yaw"]),
    )
    quat = vals[3:7] if len(vals) == 7 else [0.0, 0.0, 0.0, 1.0]
    return [
        float(base["x_base"]),
        float(base["y_base"]),
        vals[2] - float(waist_z_in_map),
        *[float(v) for v in quat],
    ]


def object_world_to_waist_live(
    object_world_pose: Sequence[float],
    node: Any,
    *,
    robot_base_world: Optional[Dict[str, Any]] = None,
    target_frame_id: Optional[str] = None,
    waist_frame: str = DEFAULT_WAIST_FRAME,
    base_frame: str = DEFAULT_BASE_FRAME,
    tf_timeout: float = 3.0,
) -> List[float]:
    """Fetch robot pose + live TF, then transform object world pose to waist."""
    if robot_base_world is None:
        robot_base_world = get_robot_pose()
    waist_from_base = waist_from_base_tf(
        node,
        waist_frame=waist_frame,
        base_frame=base_frame,
        timeout=tf_timeout,
    )
    return object_world_to_waist(
        object_world_pose,
        robot_base_world,
        waist_from_base,
        target_frame_id=target_frame_id,
    )


def planar_map_delta_to_base(target_x: float, target_y: float, robot_x: float, robot_y: float, robot_yaw: float) -> Dict[str, float]:
    """Diagnostic helper for the familiar 2D map -> base transform."""
    dx = float(target_x) - float(robot_x)
    dy = float(target_y) - float(robot_y)
    c, s = math.cos(float(robot_yaw)), math.sin(float(robot_yaw))
    return {"x_base": c * dx + s * dy, "y_base": -s * dx + c * dy, "dx_map": dx, "dy_map": dy}
