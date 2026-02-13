"""
Currency detector – extracts currency info from YOLO detections.

If the YOLO model was trained with Philippine peso bill / coin classes
(e.g. "20_peso", "100_peso", "coin_5"), this module aggregates them
into a human-readable string and computes the total.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.yolo_detector import Detection

from app.config import settings

# Map YOLO class names → (display name, numeric value)
_CURRENCY_MAP: dict[str, tuple[str, float]] = {
    "20_peso": ("₱20 bill", 20),
    "50_peso": ("₱50 bill", 50),
    "100_peso": ("₱100 bill", 100),
    "200_peso": ("₱200 bill", 200),
    "500_peso": ("₱500 bill", 500),
    "1000_peso": ("₱1000 bill", 1000),
    "coin_1": ("₱1 coin", 1),
    "coin_5": ("₱5 coin", 5),
    "coin_10": ("₱10 coin", 10),
}


def detect_currency(detections: list[Detection]) -> str | None:
    """
    Return a spoken-friendly string summarising detected currency.

    Example: "₱100 bill and ₱20 bill, total ₱120"
    Returns None if no currency is detected.
    """
    found: list[tuple[str, float]] = []

    for det in detections:
        entry = _CURRENCY_MAP.get(det.label)
        if entry is not None:
            found.append(entry)

    if not found:
        return None

    total = sum(v for _, v in found)
    names = ", ".join(n for n, _ in found)
    return f"{names} – total ₱{total:,.0f}"
