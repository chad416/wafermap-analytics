"""Raw WM-811K pickle -> cleaned, split, columnar dataset (``data/processed/wafers.parquet``).

Steps
-----
1. Load ``LSWMD.pkl`` with the restricted unpickler (no arbitrary code execution).
2. Flatten the nested label / split arrays; empty arrays mean "unlabelled".
3. Drop maps that cannot be real wafers (fewer than ``MIN_DIES`` dies, invalid bin codes).
4. Keep every wafer of every lot that has at least one labelled wafer (``scope=labelled-lots``),
   so lot drill-down shows complete lots; ``scope=all`` keeps all 811k wafers.
5. Assign a **lot-grouped**, label-stratified train/val/test split (70/15/15) to the labelled
   wafers. Wafers from one lot never straddle splits: neighbouring wafers in a lot share process
   history and look alike, so a random wafer-level split would leak and inflate test scores.
6. Write the parquet file, a manifest (raw-file hash, counts, seed) and the class-imbalance
   report + figure under ``reports/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold

from wafermap import wafer
from wafermap.constants import BIN_FAIL, BIN_NO_DIE, BIN_PASS, CLASSES, MIN_DIES
from wafermap.data.safe_pickle import load_restricted

log = logging.getLogger(__name__)

RAW_FILENAME = "LSWMD.pkl"
# SHA-256 of the file served by the Hugging Face mirror used in scripts/fetch_dataset.py.
RAW_SHA256 = "1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65"
N_FOLDS = 20  # 20 folds -> 3 test, 3 val, 14 train = 15 / 15 / 70 %
TEST_FOLDS = frozenset({0, 1, 2})
VAL_FOLDS = frozenset({3, 4, 5})

Scope = Literal["labelled-lots", "labelled", "all"]


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _flatten(cell: Any) -> str | None:
    """WM-811K stores labels as ``array([['Center']])`` or an empty array."""
    arr = np.asarray(cell)
    return str(arr.ravel()[0]) if arr.size else None


def assign_splits(labels: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Lot-grouped, label-stratified split. Returns an array of 'train' / 'val' / 'test'."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    split = np.empty(len(labels), dtype=object)
    for fold, (_, idx) in enumerate(sgkf.split(np.zeros(len(labels)), labels, groups)):
        split[idx] = "test" if fold in TEST_FOLDS else "val" if fold in VAL_FOLDS else "train"
    return split


def prepare(
    raw_dir: Path,
    processed_dir: Path,
    reports_dir: Path,
    *,
    scope: Scope = "labelled-lots",
    seed: int = 42,
    verify_hash: bool = True,
) -> dict[str, Any]:
    raw_path = raw_dir / RAW_FILENAME
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found - run `python scripts/fetch_dataset.py`")
    digest = sha256_file(raw_path)
    if verify_hash and digest != RAW_SHA256:
        raise ValueError(f"unexpected SHA-256 {digest} for {raw_path} (expected {RAW_SHA256})")

    log.info("loading %s", raw_path)
    df = load_restricted(raw_path)
    n_raw = len(df)

    lot_names = df["lotName"].astype(str).to_numpy()
    wafer_idx = df["waferIndex"].to_numpy().astype(np.int32)
    die_size = df["dieSize"].to_numpy().astype(np.int32)
    labels = np.array([_flatten(x) for x in df["failureType"]], dtype=object)
    orig_split = np.array([_flatten(x) for x in df["trianTestLabel"]], dtype=object)
    maps = df["waferMap"].tolist()
    del df

    # --- cleaning ---------------------------------------------------------------------------
    keep = np.ones(n_raw, dtype=bool)
    dropped: Counter[str] = Counter()
    dropped_labelled: Counter[str] = Counter()
    n_die = np.zeros(n_raw, dtype=np.int32)
    n_fail = np.zeros(n_raw, dtype=np.int32)
    for i, m in enumerate(maps):
        a = np.asarray(m)
        reason = None
        if a.ndim != 2 or not np.isin(a, (BIN_NO_DIE, BIN_PASS, BIN_FAIL)).all():
            reason = "invalid_bin_codes"
        else:
            n_die[i] = int(np.count_nonzero(a))
            n_fail[i] = int(np.count_nonzero(a == BIN_FAIL))
            if n_die[i] < MIN_DIES:
                reason = f"fewer_than_{MIN_DIES}_dies"
        if reason:
            keep[i] = False
            dropped[reason] += 1
            if labels[i] is not None:
                dropped_labelled[f"{reason}:{labels[i]}"] += 1

    labelled = keep & (labels != None)  # noqa: E711 - elementwise on object array
    if scope == "all":
        in_scope = keep
    elif scope == "labelled":
        in_scope = labelled
    else:
        lots_with_labels = set(lot_names[labelled])
        in_scope = keep & np.isin(lot_names, list(lots_with_labels))

    split = np.full(n_raw, None, dtype=object)
    split[labelled] = assign_splits(labels[labelled].astype(str), lot_names[labelled], seed)

    idx = np.flatnonzero(in_scope)
    log.info("encoding %d wafer maps", len(idx))
    blobs = [wafer.encode(np.asarray(maps[i], dtype=np.uint8)) for i in idx]
    shapes = np.array([np.asarray(maps[i]).shape for i in idx], dtype=np.int32)

    table = pa.table(
        {
            "lot_name": pa.array(lot_names[idx]),
            "lot_number": pa.array(
                [int(s.removeprefix("lot")) for s in lot_names[idx]], pa.int32()
            ),
            "wafer_index": pa.array(wafer_idx[idx]),
            "die_size": pa.array(die_size[idx]),
            "rows": pa.array(shapes[:, 0]),
            "cols": pa.array(shapes[:, 1]),
            "map_zlib": pa.array(blobs, pa.binary()),
            "n_die": pa.array(n_die[idx]),
            "n_fail": pa.array(n_fail[idx]),
            "label": pa.array(labels[idx].tolist(), pa.string()),
            "original_split": pa.array(orig_split[idx].tolist(), pa.string()),
            "split": pa.array(split[idx].tolist(), pa.string()),
        }
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "wafers.parquet"
    pq.write_table(table, out_path, compression="zstd")

    # --- reporting ----------------------------------------------------------------------------
    summary = build_summary(labels, split, labelled, dropped, dropped_labelled, lot_names, in_scope)
    raw_shapes = {(a.shape[0], a.shape[1]) for a in map(np.asarray, maps) if a.ndim == 2}
    summary.update(
        grid_shapes={
            "distinct_raw": len(raw_shapes),
            "distinct_in_scope": len({(int(r), int(c)) for r, c in shapes}),
            "max_rows_raw": max(r for r, _ in raw_shapes),
            "max_cols_raw": max(c for _, c in raw_shapes),
        },
        labelled_n_die_percentile_1=int(np.percentile(n_die[labelled], 1)),
        raw_file=RAW_FILENAME,
        raw_sha256=digest,
        raw_rows=n_raw,
        scope=scope,
        seed=seed,
        rows_written=len(idx),
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    # newline="\n": identical bytes on Windows and Linux
    (processed_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", newline="\n")
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report = {k: v for k, v in summary.items() if k != "created_utc"}
    (metrics_dir / "dataset_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", newline="\n"
    )
    plot_class_distribution(summary, reports_dir / "figures" / "class_distribution.png")
    log.info("wrote %s (%d rows)", out_path, len(idx))
    return summary


def build_summary(
    labels: np.ndarray,
    split: np.ndarray,
    labelled: np.ndarray,
    dropped: Counter[str],
    dropped_labelled: Counter[str],
    lot_names: np.ndarray,
    in_scope: np.ndarray,
) -> dict[str, Any]:
    lab = labels[labelled].astype(str)
    counts = Counter(lab)
    per_split = {
        s: {c: int(np.sum((split[labelled] == s) & (lab == c))) for c in CLASSES}
        for s in ("train", "val", "test")
    }
    lots_per_split = {
        s: len(set(lot_names[labelled][split[labelled] == s])) for s in ("train", "val", "test")
    }
    minority = min(counts.values())
    return {
        "labelled_wafers": int(labelled.sum()),
        "unlabelled_wafers_in_scope": int((in_scope & ~labelled).sum()),
        "lots_in_scope": len(set(lot_names[in_scope])),
        "class_counts": {c: int(counts[c]) for c in CLASSES},
        "class_share_pct": {c: round(100 * counts[c] / len(lab), 3) for c in CLASSES},
        "imbalance_ratio_max_over_min": round(max(counts.values()) / minority, 1),
        "imbalance_ratio_excluding_none": round(
            max(v for k, v in counts.items() if k != "none") / minority, 1
        ),
        "split_counts": per_split,
        "split_lots": lots_per_split,
        "dropped": dict(dropped),
        "dropped_labelled": dict(sorted(dropped_labelled.items())),
    }


def plot_class_distribution(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    counts = summary["class_counts"]
    order = sorted(CLASSES, key=lambda c: -counts[c])
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    bars = ax.bar(order, [counts[c] for c in order], color="#3b6ea8")
    ax.set_yscale("log")
    ax.set_ylabel("labelled wafers (log scale)")
    ax.set_title(
        f"WM-811K labelled subset: {summary['labelled_wafers']:,} wafers, "
        f"max/min ratio {summary['imbalance_ratio_max_over_min']:,}:1"
    )
    for b, c in zip(bars, order, strict=True):
        ax.annotate(
            f"{counts[c]:,}",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.tick_params(axis="x", rotation=30)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
