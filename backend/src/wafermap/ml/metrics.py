"""Evaluation metrics and plots shared by both models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from wafermap.constants import CLASSES

IntArray = npt.NDArray[np.int64]


def confusion(y_true: IntArray, y_pred: IntArray, n: int = len(CLASSES)) -> IntArray:
    return np.bincount(y_true * n + y_pred, minlength=n * n).reshape(n, n).astype(np.int64)


def f1_from_confusion(cm: IntArray) -> npt.NDArray[np.float64]:
    tp = np.diag(cm).astype(np.float64)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
    denom = precision + recall
    f1: npt.NDArray[np.float64] = np.divide(
        2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0
    )
    return f1


def macro_f1(y_true: IntArray, y_pred: IntArray) -> float:
    return float(f1_from_confusion(confusion(y_true, y_pred)).mean())


def bootstrap_macro_f1(
    y_true: IntArray, y_pred: IntArray, groups: npt.NDArray[Any], n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95 % CI of macro-F1 from a **lot-level** (cluster) bootstrap.

    Wafers in a lot are correlated, so resampling individual wafers would understate the
    uncertainty; we resample whole lots instead.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    n = len(CLASSES)
    # Per-lot confusion matrices, then sum lots drawn with replacement.
    per_lot = np.zeros((len(uniq), n * n), dtype=np.int64)
    np.add.at(per_lot, (inv, y_true * n + y_pred), 1)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        cm = (per_lot * counts[:, None]).sum(axis=0).reshape(n, n)
        scores[b] = f1_from_confusion(cm).mean()
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(
    y_true: IntArray, y_pred: IntArray, groups: npt.NDArray[Any] | None = None
) -> dict[str, Any]:
    cm = confusion(y_true, y_pred)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    f1 = f1_from_confusion(cm)
    per_class = {
        c: {
            "precision": round(float(tp[i] / pred_count[i]) if pred_count[i] else 0.0, 4),
            "recall": round(float(tp[i] / support[i]) if support[i] else 0.0, 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, c in enumerate(CLASSES)
    }
    out: dict[str, Any] = {
        "n": len(y_true),
        "macro_f1": round(float(f1.mean()), 4),
        "macro_f1_defect_classes": round(float(f1[:-1].mean()), 4),  # excludes 'none'
        "weighted_f1": round(float((f1 * support).sum() / support.sum()), 4),
        "accuracy": round(float(tp.sum() / cm.sum()), 4),
        "balanced_accuracy": round(float(np.mean(tp / np.maximum(support, 1))), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": list(CLASSES),
    }
    if groups is not None:
        lo, hi = bootstrap_macro_f1(y_true, y_pred, groups)
        out["macro_f1_ci95_lot_bootstrap"] = [round(lo, 4), round(hi, 4)]
    return out


def plot_confusion(cm: list[list[int]], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.asarray(cm, dtype=np.float64)
    norm = a / np.maximum(a.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if a[i, j]:
                ax.text(
                    j,
                    i,
                    f"{norm[i, j]:.2f}\n{int(a[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if norm[i, j] > 0.55 else "#1f2937",
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised (recall)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
