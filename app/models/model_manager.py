"""
ModelManager – loads and holds references to all AI models.
"""

import logging

from app.models.yolo_detector import YoloDetector
from app.models.midas_depth import MidasDepth
from app.models.zero_dce_enhancer import ZeroDCEEnhancer

logger = logging.getLogger("eyes.models")


class ModelManager:
    """Singleton-like holder for all inference models."""

    def __init__(self):
        self.yolo: YoloDetector | None = None
        self.midas: MidasDepth | None = None
        self.zero_dce: ZeroDCEEnhancer | None = None

    # ── lifecycle ──────────────────────────────────────────────

    def load_all(self):
        logger.info("Loading YOLOv8 …")
        self.yolo = YoloDetector()

        logger.info("Loading MiDaS …")
        self.midas = MidasDepth()

        logger.info("Loading Zero-DCE …")
        self.zero_dce = ZeroDCEEnhancer()

    def unload_all(self):
        self.yolo = None
        self.midas = None
        self.zero_dce = None
        logger.info("Models released.")
