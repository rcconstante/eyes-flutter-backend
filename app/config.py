"""
Central configuration – reads from environment variables with sensible defaults.
"""

import os


class Settings:
    # Model paths (relative to backend/)
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "models/yolov8n.pt")
    ZERO_DCE_MODEL_PATH: str = os.getenv("ZERO_DCE_MODEL_PATH", "models/zero_dce_model.h5")
    MIDAS_MODEL_TYPE: str = os.getenv("MIDAS_MODEL_TYPE", "MiDaS_small")

    # Processing
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "640"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    LOW_LIGHT_THRESHOLD: float = float(os.getenv("LOW_LIGHT_THRESHOLD", "0.15"))

    # Distance calibration (approximate focal-length based)
    # Reference focal length at 640px image height; auto-scaled for other resolutions
    FOCAL_LENGTH_PX: float = float(os.getenv("FOCAL_LENGTH_PX", "600.0"))
    KNOWN_HEIGHTS: dict = {
        "person": 1.7,
        "car": 1.5,
        "truck": 3.0,
        "bus": 3.2,
        "motorcycle": 1.1,
        "bicycle": 1.0,
        "dog": 0.5,
        "cat": 0.3,
        "chair": 0.9,
        "bottle": 0.25,
        "cup": 0.15,
        "cell phone": 0.14,
        "laptop": 0.3,
        "tv": 0.6,
        "door": 2.0,
        "table": 0.75,
        "bed": 0.6,
        "toilet": 0.4,
        "sink": 0.5,
        "refrigerator": 1.8,
        "oven": 0.9,
        "fire hydrant": 0.6,
        "stop sign": 0.75,
        "traffic light": 0.6,
        "bench": 0.5,
    }

    # Priority danger objects
    CRITICAL_OBJECTS: set = {
        "car", "truck", "bus", "motorcycle", "bicycle",
        "fire hydrant", "stop sign", "traffic light",
        "person", "dog", "knife", "scissors",
    }

    # Distance thresholds (metres)
    DISTANCE_VERY_CLOSE: float = 1.0
    DISTANCE_CLOSE: float = 3.0
    DISTANCE_MEDIUM: float = 5.0

    # Currency labels the YOLO model may detect
    CURRENCY_LABELS: set = {
        "20_peso", "50_peso", "100_peso", "200_peso",
        "500_peso", "1000_peso", "coin_1", "coin_5", "coin_10",
    }

    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
