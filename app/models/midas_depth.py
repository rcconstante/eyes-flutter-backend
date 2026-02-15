"""
MiDaS monocular depth estimation wrapper.

Uses the Intel/MiDaS_small model via PyTorch Hub for lightweight
depth map prediction. The depth map is used alongside YOLO bounding
boxes to estimate real-world distance to each detected object.

Two estimation strategies:
  1. **Depth-map median**: median inverse-depth value inside each bbox,
     converted to approximate metres.
  2. **Pinhole fallback**: classic focal-length / bbox-height heuristic
     when the object has a known real-world height (from config).

The final distance for each detection is the average of both when
available, otherwise whichever method succeeds.
"""

import logging

import cv2
import numpy as np
import torch
from PIL import Image

from app.config import settings

logger = logging.getLogger("eyes.midas")


class MidasDepth:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load MiDaS small via torch hub
        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            settings.MIDAS_MODEL_TYPE,
            trust_repo=True,
        )
        self.model.to(self.device).eval()

        # Matching transforms
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
        )
        if settings.MIDAS_MODEL_TYPE in ("DPT_Large", "DPT_Hybrid"):
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        logger.info(f"MiDaS ({settings.MIDAS_MODEL_TYPE}) loaded on {self.device}")

    # ── public API ────────────────────────────────────────────

    def estimate_depth_map(self, image: Image.Image) -> np.ndarray:
        """Return a depth map (H×W float32, higher = closer)."""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        input_batch = self.transform(img_cv).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_cv.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

    def estimate_distance(
        self,
        depth_map: np.ndarray,
        bbox: tuple[int, int, int, int],
        label: str,
        bbox_height_px: int,
        image_height: int,
    ) -> float:
        """
        Estimate real-world distance (metres) to the object.

        Combines depth-map based estimation and pinhole-model
        estimation when a known object height is available.
        """
        distances: list[float] = []

        # Strategy 1: Depth-map median inside bbox
        depth_dist = self._depth_map_distance(depth_map, bbox)
        if depth_dist is not None and depth_dist > 0:
            distances.append(depth_dist)

        # Strategy 2: Pinhole model
        pinhole_dist = self._pinhole_distance(label, bbox_height_px, image_height)
        if pinhole_dist is not None and pinhole_dist > 0:
            distances.append(pinhole_dist)

        if not distances:
            return 0.0

        return round(float(np.mean(distances)), 2)

    # ── internals ─────────────────────────────────────────────

    @staticmethod
    def _depth_map_distance(
        depth_map: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> float | None:
        """Convert median inverse-depth in the bbox region to metres."""
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]

        # Clamp bbox to depth map bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        # Use center 60% of bbox for more accurate depth (avoids edges)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rw = max(1, int((x2 - x1) * 0.3))
        rh = max(1, int((y2 - y1) * 0.3))
        rx1 = max(0, cx - rw)
        ry1 = max(0, cy - rh)
        rx2 = min(w, cx + rw)
        ry2 = min(h, cy + rh)

        region = depth_map[ry1:ry2, rx1:rx2]
        if region.size == 0:
            region = depth_map[y1:y2, x1:x2]

        median_val = float(np.median(region))

        if median_val <= 0:
            return None

        # MiDaS outputs relative inverse depth.
        # Normalize against the full depth map range for more stable results.
        depth_min = float(np.min(depth_map))
        depth_max = float(np.max(depth_map))
        depth_range = depth_max - depth_min
        if depth_range <= 0:
            return None

        # Normalized depth: 0 = farthest, 1 = closest
        normalized = (median_val - depth_min) / depth_range

        # Map normalized depth to distance (metres)
        # Close objects (normalized ~1.0) -> ~0.3m
        # Far objects (normalized ~0.0) -> ~15m
        MIN_DIST = 0.3
        MAX_DIST = 15.0
        if normalized <= 0.01:
            return MAX_DIST

        distance = MIN_DIST + (1.0 - normalized) * (MAX_DIST - MIN_DIST)
        return round(min(max(distance, MIN_DIST), MAX_DIST), 2)

    @staticmethod
    def _pinhole_distance(
        label: str,
        bbox_height_px: int,
        image_height: int,
    ) -> float | None:
        """Classic D = (F × H_real) / H_pixel estimate.

        Dynamically scales focal length based on image resolution
        so that distance estimates work regardless of camera resolution.
        """
        known_h = settings.KNOWN_HEIGHTS.get(label)
        if known_h is None or bbox_height_px <= 0:
            return None

        # Scale focal length proportionally to image height.
        # The configured FOCAL_LENGTH_PX assumes a 640px image height.
        # Mobile cameras send varying resolutions, so we compensate.
        REFERENCE_HEIGHT = 640.0
        scaled_focal = settings.FOCAL_LENGTH_PX * (image_height / REFERENCE_HEIGHT)

        distance = (scaled_focal * known_h) / bbox_height_px
        return round(min(max(distance, 0.2), 20.0), 2)
