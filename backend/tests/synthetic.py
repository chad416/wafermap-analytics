"""Synthetic wafer maps for tests. Not WM-811K data; used only to exercise code paths."""

from __future__ import annotations

import numpy as np

from wafermap.wafer import WaferArray


def blank(size: int = 40) -> WaferArray:
    c = (np.arange(size) + 0.5) - size / 2
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return (np.hypot(xx, yy) <= size / 2 - 0.5).astype(np.uint8)  # all pass inside the circle


def _radius(size: int) -> np.ndarray:
    c = (np.arange(size) + 0.5) - size / 2
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return np.hypot(xx, yy) / (size / 2)


def center(size: int = 40) -> WaferArray:
    m = blank(size)
    m[(_radius(size) < 0.3) & (m > 0)] = 2
    return m


def edge_ring(size: int = 40) -> WaferArray:
    m = blank(size)
    m[(_radius(size) > 0.85) & (m > 0)] = 2
    return m


def scratch(size: int = 40) -> WaferArray:
    m = blank(size)
    for i in range(size // 5, size - size // 5):
        if m[i, i]:
            m[i, i] = 2
    return m


def random_fail(size: int = 40, p: float = 0.3, seed: int = 0) -> WaferArray:
    m = blank(size)
    rng = np.random.default_rng(seed)
    m[(rng.random(m.shape) < p) & (m > 0)] = 2
    return m
