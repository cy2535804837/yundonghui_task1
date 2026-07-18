#!/usr/bin/env python3
"""Read the mobile base pose from the SLAM localization HTTP endpoint."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

HTTP_POSE_URL = os.environ.get(
    "ROBOT_HTTP_POSE_URL",
    "http://192.168.41.6:1448/api/core/slam/v1/localization/pose",
)
HTTP_TIMEOUT = float(os.environ.get("ROBOT_HTTP_POSE_TIMEOUT", "3"))
DEFAULT_FRAME_ID = os.environ.get("ROBOT_HTTP_POSE_FRAME", "map")


def _http_get_json(url: str, timeout: float) -> Dict[str, Any]:
    if not url:
        raise ValueError("robot pose URL is empty")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to GET robot pose URL {url!r}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"robot pose URL {url!r} did not return valid JSON: {raw[:200]!r}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"robot pose URL {url!r} returned non-object JSON: {type(data).__name__}")
    return data


def _float_field(data: Dict[str, Any], key: str) -> float:
    if key not in data:
        raise ValueError(f"robot pose response missing required field {key!r}: {data}")
    try:
        return float(data[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"robot pose field {key!r} must be a number, got {data[key]!r}") from exc


def get_robot_pose(
    url: str = HTTP_POSE_URL,
    timeout: float = HTTP_TIMEOUT,
    frame_id: str = DEFAULT_FRAME_ID,
    *,
    direct_x: Optional[float] = None,
    direct_y: Optional[float] = None,
    direct_yaw: Optional[float] = None,
) -> Dict[str, Any]:
    """Return the robot base pose in the SLAM map/world frame.

    Returns ``{"frame_id": str, "x": m, "y": m, "yaw": rad, ...}``.
    If ``direct_x/direct_y/direct_yaw`` are all provided, no HTTP request is made.
    """
    if direct_x is not None and direct_y is not None and direct_yaw is not None:
        return {
            "frame_id": str(frame_id),
            "x": float(direct_x),
            "y": float(direct_y),
            "yaw": float(direct_yaw),
            "source": "cli",
            "raw": None,
        }

    data = _http_get_json(url, timeout)
    return {
        "frame_id": str(data.get("frame_id") or data.get("frame") or frame_id),
        "x": _float_field(data, "x"),
        "y": _float_field(data, "y"),
        "yaw": _float_field(data, "yaw"),
        "source": "http",
        "raw": data,
    }


if __name__ == "__main__":
    print(json.dumps(get_robot_pose(), ensure_ascii=False))

