"""
ModelManager – loads and holds references to all AI models.

All imports are lazy to ensure the server starts instantly and
passes Railway health checks before heavyweight libraries load.
"""

import logging

logger = logging.getLogger("eyes.models")


class ModelManager:
    """Singleton-like holder for all inference models with lazy loading."""

    def __init__(self):
        self.yolo = None
        self.midas = None
        self.zero_dce = None

    # ── lifecycle ──────────────────────────────────────────────

    def load_all(self):
        logger.info("Loading YOLOv8 …")
        from app.models.yolo_detector import YoloDetector
        self.yolo = YoloDetector()

        logger.info("Loading MiDaS …")
        from app.models.midas_depth import MidasDepth
        self.midas = MidasDepth()

        logger.info("Loading Zero-DCE …")
        from app.models.zero_dce_enhancer import ZeroDCEEnhancer
        self.zero_dce = ZeroDCEEnhancer()

    def unload_all(self):
        self.yolo = None
        self.midas = None
        self.zero_dce = None
        logger.info("Models released.")

    # ── lazy getters ───────────────────────────────────────────

    def get_yolo(self):
        """Get YOLO detector, loading if needed."""
        if self.yolo is None:
            logger.info("Lazy loading YOLOv8 …")
            from app.models.yolo_detector import YoloDetector
            self.yolo = YoloDetector()
        return self.yolo

    def get_midas(self):
        """Get MiDaS depth estimator, loading if needed."""
        if self.midas is None:
            logger.info("Lazy loading MiDaS …")
            from app.models.midas_depth import MidasDepth
            self.midas = MidasDepth()
        return self.midas

    def get_zero_dce(self):
        """Get Zero-DCE enhancer, loading if needed."""
        if self.zero_dce is None:
            logger.info("Lazy loading Zero-DCE …")
            from app.models.zero_dce_enhancer import ZeroDCEEnhancer
            self.zero_dce = ZeroDCEEnhancer()
        return self.zero_dce
