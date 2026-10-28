"""Train the LightGBM baseline on engineered features.

Protocol
--------
* Train on ``train``, early-stop and select hyper-parameters on ``val`` (macro-F1).
* The ``test`` split is scored exactly once, after selection, and only reported.
* Class imbalance: per-sample weights ``w_c = (n_max / n_c) ** alpha``; ``alpha`` is one of the
  hyper-parameters chosen on validation (0 = no re-weighting, 1 = fully balanced).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from wafermap.constants import CLASSES
from wafermap.ml.metrics import evaluate, macro_f1, plot_confusion

log = logging.getLogger(__name__)

GRID: dict[str, list[Any]] = {
    "alpha": [0.0, 0.5, 1.0],
    "num_leaves": [31, 63],
}
BASE_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": len(CLASSES),
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 0,
}


def class_weights(y: np.ndarray, alpha: float) -> np.ndarray:
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    w = (counts.max() / np.maximum(counts, 1)) ** alpha
    return w[y]


def train_baseline(processed_dir: Path, model_dir: Path, reports_dir: Path, seed: int = 42) -> dict[str, Any]:
    d = np.load(processed_dir / "features.npz", allow_pickle=False)
    x, y, split = d["X"], d["y"], d["split"]
    names = d["feature_names"].tolist()
    lots = d["lot_name"]
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    log.info("train=%d val=%d test=%d features=%d", tr.sum(), va.sum(), te.sum(), x.shape[1])

    trials: list[dict[str, Any]] = []
    best: tuple[float, lgb.Booster, dict[str, Any]] | None = None
    for alpha, leaves in itertools.product(GRID["alpha"], GRID["num_leaves"]):
        params = {**BASE_PARAMS, "num_leaves": leaves, "seed": seed}
        dtrain = lgb.Dataset(x[tr], y[tr], weight=class_weights(y[tr], alpha), feature_name=names)
        dval = lgb.Dataset(x[va], y[va], weight=class_weights(y[va], alpha), reference=dtrain)
        t0 = time.perf_counter()
        booster = lgb.train(
            params, dtrain, num_boost_round=3000, valid_sets=[dval],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pred = booster.predict(x[va], num_iteration=booster.best_iteration).argmax(axis=1)
        score = macro_f1(y[va], pred)
        trial = {"alpha": alpha, "num_leaves": leaves, "best_iteration": booster.best_iteration,
                 "val_macro_f1": round(score, 4), "seconds": round(time.perf_counter() - t0, 1)}
        log.info("trial %s", trial)
        trials.append(trial)
        if best is None or score > best[0]:
            best = (score, booster, trial)
    assert best is not None
    _, booster, chosen = best

    val_metrics = evaluate(y[va], booster.predict(x[va], num_iteration=booster.best_iteration).argmax(1))
    # --- the single, final look at the test split ---------------------------------------------
    test_pred = booster.predict(x[te], num_iteration=booster.best_iteration).argmax(1)
    test_metrics = evaluate(y[te], test_pred, groups=lots[te])
    log.info("val macro-F1 %.4f | test macro-F1 %.4f", val_metrics["macro_f1"], test_metrics["macro_f1"])

    out_dir = model_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.txt"
    booster.save_model(model_path, num_iteration=booster.best_iteration)
    version = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    importance = dict(zip(names, booster.feature_importance("gain").round(1).tolist(), strict=True))
    meta = {
        "name": "baseline",
        "kind": "lightgbm",
        "version": version,
        "classes": list(CLASSES),
        "feature_names": names,
        "selected": chosen,
        "trials": trials,
        "val_macro_f1": val_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report = {"model": "baseline", "version": version, "selected": chosen, "trials": trials,
              "val": val_metrics, "test": test_metrics,
              "top_features_by_gain": dict(sorted(importance.items(), key=lambda kv: -kv[1])[:15])}
    (metrics_dir / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_confusion(test_metrics["confusion_matrix"], reports_dir / "figures" / "confusion_baseline.png",
                   f"LightGBM baseline - test split (n={test_metrics['n']:,}), "
                   f"macro-F1 {test_metrics['macro_f1']:.3f}")
    return report
