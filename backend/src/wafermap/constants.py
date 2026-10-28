"""Shared constants: class vocabulary and die-bin codes."""

from __future__ import annotations

from typing import Final

# WM-811K failure-pattern classes, in the fixed order used by every model output.
CLASSES: Final[tuple[str, ...]] = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
)
CLASS_TO_INDEX: Final[dict[str, int]] = {c: i for i, c in enumerate(CLASSES)}

# Die bin codes used in WM-811K wafer maps.
BIN_NO_DIE: Final[int] = 0
BIN_PASS: Final[int] = 1
BIN_FAIL: Final[int] = 2

# Canonical grid every map is resampled to for features and the CNN.
CANONICAL_SIZE: Final[int] = 64

# Sanity limits for accepted maps (WM-811K max is 300 x 205).
MIN_DIES: Final[int] = 100
MAX_SIDE: Final[int] = 512
