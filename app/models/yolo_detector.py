"""
YOLOv8 object detection wrapper.

Uses the Ultralytics YOLOv8n model (same architecture as the training script).
On first run it downloads the pretrained weights if not found locally.
During deployment you should upload your custom-trained `yolov8n.pt` to the
`models/` directory.
"""

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.config import settings

logger = logging.getLogger("eyes.yolo")


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    bbox_height_px: int


class YoloDetector:
    def __init__(self):
        self.model = YOLO(settings.YOLO_MODEL_PATH)
        logger.info(f"YOLO loaded from {settings.YOLO_MODEL_PATH}")

    def detect(self, image: Image.Image) -> list[Detection]:
        """Run inference on a PIL Image and return a list of Detections."""
        results = self.model.predict(
            source=image,
            imgsz=settings.IMAGE_SIZE,
            conf=settings.CONFIDENCE_THRESHOLD,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                bbox_h = int(y2 - y1)
                detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        bbox_height_px=bbox_h,
                    )
                )
        return detections
