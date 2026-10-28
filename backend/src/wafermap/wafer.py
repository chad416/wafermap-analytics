"""Wafer-map primitives: validation, statistics, compact storage and canonical resampling.

A wafer map is a 2-D ``uint8`` array where 0 = no die, 1 = pass, 2 = fail (WM-811K coding).
"""

from __future__ import annotations

import io
import json
import zlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from wafermap.constants import BIN_FAIL, BIN_NO_DIE, BIN_PASS, CANONICAL_SIZE, MAX_SIDE, MIN_DIES

WaferArray = npt.NDArray[np.uint8]


class InvalidWaferMap(ValueError):
    """Raised when an input cannot be interpreted as a wafer map."""


@dataclass(frozen=True, slots=True)
class WaferStats:
    n_die: int
    n_pass: int
    n_fail: int

    @property
    def yield_pct(self) -> float:
        return 100.0 * self.n_pass / self.n_die if self.n_die else 0.0


def validate(arr: npt.ArrayLike, *, min_dies: int = MIN_DIES) -> WaferArray:
    """Return ``arr`` as a validated ``uint8`` wafer map or raise :class:`InvalidWaferMap`."""
    a = np.asarray(arr)
    if a.ndim != 2:
        raise InvalidWaferMap(f"wafer map must be 2-D, got {a.ndim}-D")
    if a.shape[0] < 3 or a.shape[1] < 3:
        raise InvalidWaferMap(f"wafer map too small: {a.shape}")
    if a.shape[0] > MAX_SIDE or a.shape[1] > MAX_SIDE:
        raise InvalidWaferMap(f"wafer map too large: {a.shape} (max side {MAX_SIDE})")
    if not np.issubdtype(a.dtype, np.number):
        raise InvalidWaferMap("wafer map must be numeric")
    if np.issubdtype(a.dtype, np.floating) and not np.all(np.isfinite(a)):
        raise InvalidWaferMap("wafer map contains NaN/inf")
    if not np.all(np.isin(a, (BIN_NO_DIE, BIN_PASS, BIN_FAIL))):
        raise InvalidWaferMap("wafer map values must be 0 (no die), 1 (pass) or 2 (fail)")
    out = a.astype(np.uint8)
    n_die = int(np.count_nonzero(out))
    if n_die < min_dies:
        raise InvalidWaferMap(f"wafer map has {n_die} dies, need at least {min_dies}")
    return out


def stats(m: WaferArray) -> WaferStats:
    n_pass = int(np.count_nonzero(m == BIN_PASS))
    n_fail = int(np.count_nonzero(m == BIN_FAIL))
    return WaferStats(n_die=n_pass + n_fail, n_pass=n_pass, n_fail=n_fail)


# --- compact storage -------------------------------------------------------------------------
# Two bits per die would be smaller, but zlib on the raw uint8 grid already gets ~20x on
# WM-811K maps (they are mostly runs of 0/1) and keeps decoding trivial.


def encode(m: WaferArray) -> bytes:
    return zlib.compress(np.ascontiguousarray(m, dtype=np.uint8).tobytes(), level=6)


def decode(blob: bytes, rows: int, cols: int) -> WaferArray:
    flat = np.frombuffer(zlib.decompress(blob), dtype=np.uint8)
    if flat.size != rows * cols:
        raise InvalidWaferMap(f"blob holds {flat.size} values, expected {rows}x{cols}")
    return flat.reshape(rows, cols).copy()


# --- parsing uploads ---------------------------------------------------------------------------


def parse_upload(data: bytes, filename: str) -> WaferArray:
    """Parse an uploaded map. Supports ``.npy``, ``.json`` (2-D list) and CSV/TXT grids."""
    name = filename.lower()
    try:
        if name.endswith(".npy"):
            arr = np.load(io.BytesIO(data), allow_pickle=False)
        elif name.endswith(".json"):
            payload = json.loads(data.decode("utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("grid", payload.get("wafer_map"))
            arr = np.asarray(payload)
        elif name.endswith((".csv", ".txt")):
            text = data.decode("utf-8").strip()
            rows = [r.replace(",", " ").split() for r in text.splitlines() if r.strip()]
            if len({len(r) for r in rows}) != 1:
                raise InvalidWaferMap("rows have different lengths")
            arr = np.asarray(rows, dtype=np.int64)
        else:
            raise InvalidWaferMap("unsupported file type; use .npy, .json, .csv or .txt")
    except InvalidWaferMap:
        raise
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidWaferMap(f"could not parse {filename}: {exc}") from exc
    return validate(arr)


# --- canonical resampling ------------------------------------------------------------------------


def _area_resize(img: npt.NDArray[np.float32], size: int) -> npt.NDArray[np.float32]:
    """Exact area-average resampling of a 2-D array to ``size x size`` via an integral image.

    Each output cell averages the (fractional) input area it covers, so a fail fraction is
    preserved regardless of the original die grid, and upsampling small maps is exact too.
    """
    h, w = img.shape
    # Integral image with a zero row/col; integrate along each axis with linear interpolation
    # so fractional boundaries are handled exactly.
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)
    ys = np.linspace(0.0, h, size + 1)
    xs = np.linspace(0.0, w, size + 1)

    def interp_rows(table: npt.NDArray[np.float64], pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        lo = np.floor(pts).astype(int)
        lo = np.clip(lo, 0, table.shape[0] - 2)
        frac = (pts - lo)[:, None]
        return table[lo] * (1 - frac) + table[lo + 1] * frac

    # Piecewise-linear interpolation of the integral image is exact for piecewise-constant input.
    t = interp_rows(ii, ys)  # (size+1, w+1)
    t = interp_rows(t.T, xs).T  # (size+1, size+1)
    area = (h / size) * (w / size)
    cells = t[1:, 1:] - t[:-1, 1:] - t[1:, :-1] + t[:-1, :-1]
    return (cells / area).astype(np.float32)


def canonical(m: WaferArray, size: int = CANONICAL_SIZE) -> npt.NDArray[np.float32]:
    """Resample a map to a ``(2, size, size)`` float tensor: die coverage and fail coverage."""
    die = (m != BIN_NO_DIE).astype(np.float32)
    fail = (m == BIN_FAIL).astype(np.float32)
    return np.stack([_area_resize(die, size), _area_resize(fail, size)]).clip(0.0, 1.0)


def canonical_u8(m: WaferArray, size: int = CANONICAL_SIZE) -> npt.NDArray[np.uint8]:
    """:func:`canonical` quantised to uint8 - the exact CNN input representation (x / 255)."""
    return np.rint(canonical(m, size) * 255.0).astype(np.uint8)
