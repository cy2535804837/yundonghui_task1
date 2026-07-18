from __future__ import annotations

import base64
from typing import List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


class PoseEstimator:
    def __init__(
        self,
        intrinsic_matrix: Optional[np.ndarray] = None,
        extrinsic_matrix: Optional[np.ndarray] = None,
    ):
        if intrinsic_matrix is None:
            intrinsic_matrix = np.array(
                [[612.791, 0.0, 321.736], [0.0, 611.878, 245.066], [0.0, 0.0, 1.0]]
            )
        if extrinsic_matrix is None:
            extrinsic_matrix = np.eye(4)
        self.intrinsic_matrix = intrinsic_matrix
        self.extrinsic_matrix = extrinsic_matrix
        self.inv_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)

    @classmethod
    def from_yaml(cls, yaml_file_path: str) -> "PoseEstimator":
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        intrinsic_data = config.get("camera", {}).get("intrinsic", {})
        intrinsic_matrix = np.array(
            [
                [intrinsic_data.get("fx", 612.791), 0.0, intrinsic_data.get("cx", 321.736)],
                [0.0, intrinsic_data.get("fy", 611.878), intrinsic_data.get("cy", 245.066)],
                [0.0, 0.0, 1.0],
            ]
        )
        extrinsic_data = config.get("camera", {}).get("extrinsic", {})
        extrinsic_matrix = np.eye(4)
        if "matrix" in extrinsic_data:
            extrinsic_matrix = np.array(extrinsic_data["matrix"]).reshape(4, 4)
        elif "rotation" in extrinsic_data and "translation" in extrinsic_data:
            rotation = np.array(extrinsic_data["rotation"]).reshape(3, 3)
            translation = np.array(extrinsic_data["translation"]).reshape(3, 1)
            extrinsic_matrix[:3, :3] = rotation
            extrinsic_matrix[:3, 3] = translation.flatten()
        return cls(intrinsic_matrix, extrinsic_matrix)

    def mask_to_point_cloud(
        self, mask: np.ndarray, depth_image: np.ndarray, edge_margin_ratio=0.1
    ) -> np.ndarray:
        depth_image_modified = depth_image.astype(np.float32)
        h, w = depth_image_modified.shape[:2]
        mx, my = int(w * edge_margin_ratio), int(h * edge_margin_ratio)
        if mx > 0:
            depth_image_modified[:, :mx] = np.nan
            depth_image_modified[:, w - mx :] = np.nan
        if my > 0:
            depth_image_modified[:my, :] = np.nan
            depth_image_modified[h - my :, :] = np.nan
        valid_mask = mask >= 125
        py, px = np.where(valid_mask)
        pz = depth_image_modified[py, px]
        ok = ~np.isnan(pz) & (pz > 0)
        px, py, pz = px[ok], py[ok], pz[ok]
        if len(pz) == 0:
            return np.array([]).reshape(0, 3)
        pix = np.stack([px, py, np.ones_like(px)], axis=1)
        cam = (self.inv_intrinsic_matrix @ pix.T).T
        cam *= pz[:, np.newaxis]
        cam_h = np.hstack([cam, np.ones((cam.shape[0], 1))])
        world_h = (self.extrinsic_matrix @ cam_h.T).T
        return world_h[:, :3]

    def pixel_to_3d_point(self, pixel_coords: List[float], depth_image: np.ndarray) -> np.ndarray:
        x, y = pixel_coords
        u, v = int(round(x)), int(round(y))
        if u < 0 or u >= depth_image.shape[1] or v < 0 or v >= depth_image.shape[0]:
            return np.array([np.nan, np.nan, np.nan])
        depth_value = depth_image[v, u]
        if depth_value <= 0:
            return np.array([np.nan, np.nan, np.nan])
        pixel_coord = np.array([u, v, 1.0])
        cam_coord = depth_value * np.dot(self.inv_intrinsic_matrix, pixel_coord)
        cam_h = np.append(cam_coord, 1.0)
        world_h = np.dot(self.extrinsic_matrix, cam_h)
        return world_h[:3]

    def compute_orientation_from_point_cloud(self, points_3d):
        centroid = np.mean(points_3d, axis=0)
        centered_points = points_3d - centroid
        covariance_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] = -eigenvectors[:, 2]
        rotation = Rotation.from_matrix(eigenvectors)
        qx, qy, qz, qw = rotation.as_quat()
        return qx, qy, qz, qw

    def compute_grasp_axes_from_point_cloud(self, points_3d) -> Optional[dict]:
        if len(points_3d) < 3:
            return None
        centered = points_3d - np.mean(points_3d, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        return {
            "minor_axis": eigenvectors[:, 0].tolist(),
            "medium_axis": eigenvectors[:, 1].tolist(),
            "major_axis": eigenvectors[:, 2].tolist(),
            "eigenvalues": eigenvalues.tolist(),
        }

    def estimate_pose(self, mask: np.ndarray, depth_image: np.ndarray) -> Tuple[np.ndarray, dict]:
        points_3d = self.mask_to_point_cloud(mask, depth_image)
        if len(points_3d) < 10:
            return np.array([0, 0, 0, 0, 0, 0, 1]), {"point_count": len(points_3d)}
        median = np.median(points_3d, axis=0)
        diff = np.linalg.norm(points_3d - median, axis=1)
        points_3d = points_3d[diff < np.percentile(diff, 80)]
        if len(points_3d) == 0:
            return np.array([0, 0, 0, 0, 0, 0, 1]), {"point_count": 0}
        centroid = np.mean(points_3d, axis=0)
        qx, qy, qz, qw = self.compute_orientation_from_point_cloud(points_3d)
        pose = np.array([centroid[0], centroid[1], centroid[2], qx, qy, qz, qw])
        return pose, {
            "point_count": len(points_3d),
            "centroid": centroid,
            "points_3d": points_3d,
            "grasp_axes": self.compute_grasp_axes_from_point_cloud(points_3d),
        }

    def compute_bounding_box(self, points_3d: np.ndarray) -> dict:
        if len(points_3d) == 0:
            z = np.zeros(3)
            return {"min": z, "max": z, "center": z, "size": z}
        mn = np.min(points_3d, axis=0)
        mx = np.max(points_3d, axis=0)
        return {"min": mn, "max": mx, "center": (mn + mx) / 2, "size": mx - mn}


def _decode_mask_field(mask_field) -> np.ndarray:
    if isinstance(mask_field, dict):
        enc = str(mask_field.get("encoding", "")).lower().strip()
        if enc == "packbits_b64":
            shp = mask_field.get("shape")
            if not isinstance(shp, (list, tuple)) or len(shp) < 2:
                raise ValueError(f"invalid packbits mask shape: {shp!r}")
            h, w = int(shp[0]), int(shp[1])
            raw = base64.b64decode(mask_field["data"])
            n = h * w
            bits = np.unpackbits(
                np.frombuffer(raw, dtype=np.uint8), bitorder="little"
            )[:n]
            return bits.reshape(h, w).astype(np.float32)
        raise ValueError(f"unsupported mask dict encoding: {mask_field.get('encoding')!r}")
    mask = np.array(mask_field)
    if mask.ndim == 3:
        mask = mask[0]
    elif mask.ndim == 1:
        side = int(np.sqrt(len(mask)))
        if side * side != len(mask):
            raise ValueError(f"1D mask length is not square: {len(mask)}")
        mask = mask.reshape((side, side))
    if mask.max() > 1:
        mask = mask / 255.0
    return mask.astype(np.float32)


def extract_masks_from_results(results: dict, depth_image: np.ndarray):
    """Return one entry per detection, index-aligned with results["results"].

    Failed/missing masks are returned as ``None`` (not dropped) so callers can
    keep label/bbox correspondence with ``results["results"][i]``.
    """
    if "results" not in results or not results["results"]:
        print("Warning: No segmentation results returned from API or 'results' key missing")
        return []
    masks = []
    from PIL import Image

    h, w = depth_image.shape[:2]
    for detection in results["results"]:
        if "mask" not in detection:
            print("Warning: 'mask' key not found in detection result")
            masks.append(None)
            continue
        try:
            mask = _decode_mask_field(detection["mask"])
        except Exception as e:
            print(f"Warning: failed to decode mask payload, skip one object: {e}")
            masks.append(None)
            continue
        mask_resized = np.array(
            Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h))
        )
        binary_mask = (mask_resized > 127).astype(np.uint8) * 255
        masks.append(binary_mask)
    return masks

