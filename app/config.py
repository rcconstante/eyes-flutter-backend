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
    # Formula: distance (m) = KNOWN_HEIGHT (m) * FOCAL_LENGTH_PX / bbox_height_px
    FOCAL_LENGTH_PX: float = float(os.getenv("FOCAL_LENGTH_PX", "600.0"))

    # Real-world heights (metres) for all 80 COCO classes detected by YOLOv8n.
    # Values represent the dominant visible vertical dimension of each object.
    KNOWN_HEIGHTS: dict = {
        # ── People & Riding ──────────────────────────────────────────────────
        "person":           1.7,   # average adult standing height
        "bicycle":          1.0,   # handlebar height
        "motorcycle":       1.1,   # seat + handlebar height
        "skateboard":       0.15,  # deck-to-ground height (ground-level hazard)
        "surfboard":        1.8,   # board length (usually propped upright)
        "skis":             1.6,   # ski length (upright)
        "snowboard":        1.4,   # board length (upright)

        # ── Vehicles ─────────────────────────────────────────────────────────
        "car":              1.5,   # roof height
        "truck":            3.0,   # cab roof height
        "bus":              3.2,   # full vehicle height
        "train":            3.5,   # carriage height
        "airplane":         5.0,   # fuselage height (ground level)
        "boat":             2.0,   # typical small vessel freeboard + deckhouse

        # ── Traffic Infrastructure ───────────────────────────────────────────
        "traffic light":    0.6,   # signal head box height
        "fire hydrant":     0.6,   # hydrant body height
        "stop sign":        0.75,  # sign panel height
        "parking meter":    1.2,   # meter post height

        # ── Outdoor Furniture & Fixtures ─────────────────────────────────────
        "bench":            0.5,   # seat height

        # ── Animals ──────────────────────────────────────────────────────────
        "bird":             0.2,   # small-to-medium bird (e.g. pigeon, crow)
        "cat":              0.3,   # shoulder height
        "dog":              0.5,   # shoulder height (medium breed average)
        "horse":            1.6,   # wither height
        "sheep":            0.9,   # shoulder height
        "cow":              1.4,   # shoulder height
        "elephant":         3.0,   # shoulder height
        "bear":             1.2,   # shoulder height (standing ~2 m; use shoulder)
        "zebra":            1.5,   # shoulder height
        "giraffe":          5.0,   # full height

        # ── Accessories & Bags ───────────────────────────────────────────────
        "backpack":         0.45,  # bag height (worn or on ground)
        "umbrella":         0.9,   # folded/closed length
        "handbag":          0.3,   # bag height
        "tie":              1.4,   # knot-to-tip length (hanging)
        "suitcase":         0.7,   # upright suitcase height

        # ── Sports Equipment ─────────────────────────────────────────────────
        "frisbee":          0.27,  # diameter
        "sports ball":      0.22,  # average (soccer ball ≈ 22 cm diameter)
        "kite":             0.8,   # approximate visible height when airborne
        "baseball bat":     0.85,  # length
        "baseball glove":   0.25,  # glove height
        "tennis racket":    0.68,  # racket length

        # ── Food & Kitchen (close-range, indoor use) ─────────────────────────
        "bottle":           0.25,  # typical water/beverage bottle
        "wine glass":       0.22,  # glass height
        "cup":              0.15,  # mug/cup height
        "fork":             0.20,  # fork length
        "knife":            0.25,  # kitchen knife length
        "spoon":            0.18,  # spoon length
        "bowl":             0.10,  # bowl height
        "banana":           0.18,  # banana length
        "apple":            0.08,  # apple diameter
        "sandwich":         0.12,  # sandwich height
        "orange":           0.08,  # orange diameter
        "broccoli":         0.20,  # head height
        "carrot":           0.18,  # carrot length
        "hot dog":          0.15,  # hot dog length
        "pizza":            0.30,  # pizza diameter (single slice ~0.15 m)
        "donut":            0.10,  # donut diameter
        "cake":             0.15,  # cake height

        # ── Indoor Furniture ─────────────────────────────────────────────────
        "chair":            0.90,  # seat-back height
        "couch":            0.85,  # seat-back height
        "potted plant":     0.40,  # average indoor plant height
        "bed":              0.60,  # mattress + frame height
        "dining table":     0.75,  # table surface height
        "toilet":           0.40,  # bowl height
        "tv":               0.60,  # screen height (40–55 inch typical)
        "laptop":           0.30,  # open-lid height
        "mouse":            0.04,  # mouse body height
        "remote":           0.18,  # remote length
        "keyboard":         0.03,  # keyboard thickness (keycap height)
        "cell phone":       0.15,  # phone height
        "microwave":        0.30,  # oven height
        "oven":             0.90,  # full oven height
        "toaster":          0.20,  # toaster height
        "sink":             0.50,  # basin depth / mounting height
        "refrigerator":     1.80,  # fridge height
        "clock":            0.30,  # wall-clock diameter
        "vase":             0.30,  # vase height
        "scissors":         0.18,  # scissors length
        "teddy bear":       0.30,  # stuffed toy height
        "hair drier":       0.25,  # dryer length
        "toothbrush":       0.18,  # toothbrush length
        "book":             0.25,  # book height (A5)
    }

    # ── Priority tiers for the safety engine ─────────────────────────────────
    # Tier 1: Imminent physical danger – always surfaces as the priority object.
    CRITICAL_OBJECTS: set = {
        # Moving vehicles (collision hazard)
        "car", "truck", "bus", "motorcycle", "bicycle", "train",
        # Traffic infrastructure (navigation hazard)
        "traffic light", "stop sign", "fire hydrant", "parking meter",
        # Living beings that can move unpredictably
        "person", "dog", "horse", "cow", "elephant", "bear",
        # Handheld weapons / sharp objects
        "knife", "scissors", "baseball bat",
    }

    # Tier 2: Trip / collision hazard – elevated priority when very close.
    HAZARD_OBJECTS: set = {
        "bench", "potted plant", "suitcase", "backpack",
        "skateboard", "sports ball", "bird", "cat", "sheep", "zebra",
        "couch", "chair", "dining table",
    }

    # Distance thresholds (metres)
    DISTANCE_VERY_CLOSE: float = 1.0   # strong haptic + urgent voice
    DISTANCE_CLOSE: float = 3.0        # moderate haptic + normal voice
    DISTANCE_MEDIUM: float = 5.0       # soft haptic only

    # Currency labels the YOLO model may detect
    CURRENCY_LABELS: set = {
        "20_peso", "50_peso", "100_peso", "200_peso",
        "500_peso", "1000_peso", "coin_1", "coin_5", "coin_10",
    }

    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
