"""Compute engineered features for the labelled subset -> ``data/processed/features.npz``."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
from joblib import Parallel, delayed

from wafermap import wafer
from wafermap.constants import CLASS_TO_INDEX
from wafermap.features.engineered import N_FEATURES, extract, feature_names

log = logging.getLogger(__name__)


def _chunk(blobs: list[bytes], rows: list[int], cols: list[int]) -> npt.NDArray[np.float32]:
    out = np.empty((len(blobs), N_FEATURES), dtype=np.float32)
    for i, (b, r, c) in enumerate(zip(blobs, rows, cols, strict=True)):
        out[i] = extract(wafer.decode(b, r, c))
    return out


def canonical_chunk(blobs: list[bytes], rows: list[int], cols: list[int]) -> npt.NDArray[np.uint8]:
    """uint8 canonical tensors for a chunk of encoded maps (joblib worker; no torch import)."""
    return np.stack(
        [
            wafer.canonical_u8(wafer.decode(b, r, c))
            for b, r, c in zip(blobs, rows, cols, strict=True)
        ]
    )


def load_labelled(processed_dir: Path) -> dict[str, np.ndarray]:
    """Load the labelled rows of the processed parquet as numpy arrays."""
    t = pq.read_table(
        processed_dir / "wafers.parquet",
        columns=["lot_name", "wafer_index", "rows", "cols", "map_zlib", "label", "split"],
        filters=[("split", "!=", "")],  # null split = unlabelled -> excluded by the filter
    )
    d = t.to_pydict()
    return {
        "lot_name": np.asarray(d["lot_name"]),
        "wafer_index": np.asarray(d["wafer_index"], dtype=np.int32),
        "rows": np.asarray(d["rows"], dtype=np.int32),
        "cols": np.asarray(d["cols"], dtype=np.int32),
        "map_zlib": np.asarray(d["map_zlib"], dtype=object),
        "y": np.asarray([CLASS_TO_INDEX[x] for x in d["label"]], dtype=np.int64),
        "split": np.asarray(d["split"]),
    }


def build_feature_table(processed_dir: Path, n_jobs: int = -1, chunk_size: int = 2000) -> Path:
    data = load_labelled(processed_dir)
    n = len(data["y"])
    log.info("extracting %d features for %d wafers", N_FEATURES, n)
    t0 = time.perf_counter()
    chunks = [slice(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
    parts = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_chunk)(
            data["map_zlib"][s].tolist(), data["rows"][s].tolist(), data["cols"][s].tolist()
        )
        for s in chunks
    )
    x = np.concatenate(parts)
    log.info("features done in %.1f s", time.perf_counter() - t0)
    out = processed_dir / "features.npz"
    np.savez_compressed(
        out,
        X=x,
        y=data["y"],
        split=data["split"],
        lot_name=data["lot_name"],
        wafer_index=data["wafer_index"],
        feature_names=np.asarray(feature_names()),
    )
    return out
