"""Paired comparison of the two committed models on the lot-grouped test split.

Re-runs inference with the committed weights in ``models/`` and checks that the predictions
reproduce the confusion matrices stored in ``reports/metrics/{baseline,cnn}.json``. Then:

* a paired lot-level bootstrap of the macro-F1 difference (CNN minus baseline);
* top-label calibration (expected calibration error, 15 bins) of each model;
* the largest off-diagonal confusions of each model.

The models are frozen and nothing is selected here, so scoring the test split again adds no
tuning. Needs ``data/processed/`` from ``wafermap prepare`` and ``wafermap features``.

    cd backend && uv run python ../scripts/compare_models.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import onnxruntime as ort

from wafermap.constants import CLASSES
from wafermap.features.build import canonical_chunk, load_labelled
from wafermap.ml.metrics import (
    confusion,
    evaluate,
    expected_calibration_error,
    paired_bootstrap_macro_f1_diff,
)
from wafermap.ml.predict import softmax

ROOT = Path(__file__).resolve().parents[1]


def top_confusions(cm: np.ndarray, k: int = 6) -> list[dict[str, Any]]:
    support = cm.sum(axis=1)
    cells = [(int(cm[i, j]), i, j) for i in range(len(CLASSES)) for j in range(len(CLASSES))]
    off = sorted((c for c in cells if c[1] != c[2] and c[0] > 0), reverse=True)[:k]
    return [
        {
            "true": CLASSES[i],
            "predicted": CLASSES[j],
            "count": n,
            "share_of_true_class": round(n / support[i], 4),
        }
        for n, i, j in off
    ]


def main() -> None:
    processed = ROOT / "data" / "processed"
    feats = np.load(processed / "features.npz", allow_pickle=False)
    te = feats["split"] == "test"
    y, lots = feats["y"][te], feats["lot_name"][te]

    booster = lgb.Booster(model_file=str(ROOT / "models" / "baseline" / "model.txt"))
    proba_base = np.asarray(booster.predict(feats["X"][te]), dtype=np.float32)

    data = load_labelled(processed)
    idx = np.flatnonzero(data["split"] == "test")
    assert np.array_equal(data["y"][idx], y), "features.npz and wafers.parquet disagree"
    x = canonical_chunk(
        data["map_zlib"][idx].tolist(), data["rows"][idx].tolist(), data["cols"][idx].tolist()
    )
    sess = ort.InferenceSession(
        str(ROOT / "models" / "cnn" / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    logits = np.concatenate(
        [
            sess.run(None, {"x": x[i : i + 1024].astype(np.float32) / 255.0})[0]
            for i in range(0, len(x), 1024)
        ]
    )
    proba_cnn = softmax(logits)

    out: dict[str, Any] = {"test_wafers": int(te.sum()), "test_lots": len(set(lots))}
    preds = {"baseline": proba_base.argmax(1), "cnn": proba_cnn.argmax(1)}
    probas = {"baseline": proba_base, "cnn": proba_cnn}
    for name, pred in preds.items():
        stored = json.loads((ROOT / "reports" / "metrics" / f"{name}.json").read_text())["test"]
        cm = confusion(y, pred)
        out[name] = {
            "macro_f1": evaluate(y, pred)["macro_f1"],
            "reproduces_stored_confusion_matrix": cm.tolist() == stored["confusion_matrix"],
            "calibration": expected_calibration_error(probas[name], y),
            "top_confusions": top_confusions(cm),
        }
    out["cnn_minus_baseline_macro_f1"] = paired_bootstrap_macro_f1_diff(
        y, preds["baseline"], preds["cnn"], lots
    )

    path = ROOT / "reports" / "metrics" / "comparison.json"
    path.write_text(json.dumps(out, indent=2) + "\n", newline="\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
