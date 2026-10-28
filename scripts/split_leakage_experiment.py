"""How much does a random (wafer-level) split inflate the score versus a lot-grouped split?

Trains the *selected* baseline configuration (from models/baseline/meta.json) on a random,
label-stratified 70/15/15 split of the same labelled wafers, with the same early-stopping
protocol, and compares its test macro-F1 with the lot-grouped result in reports/metrics/baseline.json.

The random-split test set is a different sample, so the comparison is between protocols,
not a paired test.

    cd backend && uv run python ../scripts/split_leakage_experiment.py
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from wafermap.ml.metrics import evaluate
from wafermap.ml.train_baseline import BASE_PARAMS, class_weights

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    d = np.load(ROOT / "data" / "processed" / "features.npz", allow_pickle=False)
    x, y, lots = d["X"], d["y"], d["lot_name"]
    meta = json.loads((ROOT / "models" / "baseline" / "meta.json").read_text())
    sel = meta["selected"]
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=42)
    shared_lots = len(set(lots[tr]) & set(lots[te]))

    params = {**BASE_PARAMS, "num_leaves": sel["num_leaves"], "seed": 42}
    dtrain = lgb.Dataset(x[tr], y[tr], weight=class_weights(y[tr], sel["alpha"]))
    dval = lgb.Dataset(x[va], y[va], weight=class_weights(y[va], sel["alpha"]), reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = np.asarray(booster.predict(x[te], num_iteration=booster.best_iteration)).argmax(1)
    random_metrics = evaluate(y[te], pred, groups=lots[te])

    grouped = json.loads((ROOT / "reports" / "metrics" / "baseline.json").read_text())["test"]
    out = {
        "config": sel,
        "random_split": {
            "test_macro_f1": random_metrics["macro_f1"],
            "ci95": random_metrics["macro_f1_ci95_lot_bootstrap"],
            "test_lots_also_in_train": shared_lots,
            "test_lots": int(len(set(lots[te]))),
        },
        "lot_grouped_split": {
            "test_macro_f1": grouped["macro_f1"],
            "ci95": grouped["macro_f1_ci95_lot_bootstrap"],
            "test_lots_also_in_train": 0,
        },
        "inflation_macro_f1": round(random_metrics["macro_f1"] - grouped["macro_f1"], 4),
    }
    path = ROOT / "reports" / "metrics" / "split_leakage.json"
    path.write_text(json.dumps(out, indent=2) + "\n", newline="\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
