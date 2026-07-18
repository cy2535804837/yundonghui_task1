from __future__ import annotations

import base64
from io import BytesIO
from typing import List

import numpy as np
import requests
from PIL import Image

from .pose_estimator import PoseEstimator, extract_masks_from_results


class PerceptionTool:
    def __init__(self, base_url: str = "http://localhost:8008", camera_pose_file_path: str = ""):
        self.base_url = base_url.rstrip("/")
        self.estimator = PoseEstimator.from_yaml(camera_pose_file_path)
        self.health_check()

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=(5, 5))
            ok = response.status_code == 200 and response.json().get("status") == "healthy"
            if ok:
                print("perception health check pass")
            else:
                print(f"[PerceptionTool] health check returned status={response.status_code}")
            return ok
        except requests.exceptions.ConnectionError as e:
            print(f"[PerceptionTool] health check FAILED - cannot connect to {self.base_url}: {e}")
            return False
        except requests.exceptions.Timeout:
            print(f"[PerceptionTool] health check FAILED - timed out connecting to {self.base_url}")
            return False
        except Exception as e:
            print(f"[PerceptionTool] health check FAILED - {type(e).__name__}: {e}")
            return False

    def segment_multi_target_image(self, encoded_image, prompts: List[str], confidence=0.8):
        data = {
            "image_data": encoded_image,
            "prompts": prompts,
            "confidence_threshold": confidence,
        }
        try:
            response = requests.post(
                f"{self.base_url}/segment/multi_target",
                json=data,
                timeout=(10, 120),  # (connect_timeout, read_timeout)
            )
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"[PerceptionTool] cannot connect to segmentation server at "
                f"{self.base_url} - is the FastSeg server running? ({e})"
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"[PerceptionTool] timed out waiting for segmentation server at "
                f"{self.base_url} (connect 10s / read 120s)"
            ) from e
        if response.status_code == 200:
            return response.json()
        raise Exception(f"API调用失败: {response.text}")

    def preprocess(self, rgb_image: np.ndarray) -> str:
        image_ori = Image.fromarray(rgb_image.astype("uint8"))
        if image_ori.mode != "RGB":
            image_ori = image_ori.convert("RGB")
        buffered = BytesIO()
        fmt = str(__import__("os").environ.get("SAM3_UPLOAD_FORMAT", "jpeg")).lower().strip()
        if fmt == "png":
            image_ori.save(buffered, format="PNG", optimize=True)
        else:
            q = int(__import__("os").environ.get("SAM3_JPEG_QUALITY", "85"))
            q = max(1, min(95, q))
            image_ori.save(
                buffered,
                format="JPEG",
                quality=q,
                optimize=True,
                subsampling=2,
            )
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    def perception_pipeline(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        prompts: List[str],
        confidence: float = 0.8,
    ):
        encoded_image = self.preprocess(rgb_image)
        try:
            results = self.segment_multi_target_image(
                encoded_image, prompts, confidence=float(confidence)
            )
        except Exception as e:
            print(f"错误: 分割 API 调用失败: {e}")
            return None, {"error": str(e)}
        if not isinstance(results, dict) or "results" not in results:
            return None, results
        if not results.get("results"):
            return None, results

        masks = extract_masks_from_results(results, depth_image)
        results_output = []
        if len(masks) == 0:
            return None, None
        for i, mask in enumerate(masks):
            if mask is None:
                continue
            pose, info = self.estimator.estimate_pose(mask, depth_image)
            if "points_3d" not in info:
                continue
            bbox_3d = self.estimator.compute_bounding_box(info["points_3d"])
            results_output.append(
                {
                    "class_name": results["results"][i]["label"],
                    "pose": pose,
                    "bbox_2d": results["results"][i]["bbox"],
                    "bbox_3d": bbox_3d,
                    "mask_id": i,
                    "grasp_axes": info.get("grasp_axes"),
                }
            )
        return (results_output if results_output else None), results

