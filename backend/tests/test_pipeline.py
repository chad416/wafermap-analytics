"""Data-pipeline and ML utility tests (no dataset needed)."""

from __future__ import annotations

import io
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from synthetic import blank, center, edge_ring, scratch
from wafermap.data.prepare import _flatten, assign_splits
from wafermap.data.safe_pickle import RestrictedUnpickler
from wafermap.features.engineered import N_FEATURES, extract, feature_names
from wafermap.ml.metrics import bootstrap_macro_f1, confusion, evaluate, macro_f1


class _Evil:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (os.system, ("echo pwned",))


def test_restricted_unpickler_blocks_code_execution() -> None:
    with pytest.raises(pickle.UnpicklingError, match="not allow-listed"):
        RestrictedUnpickler(io.BytesIO(pickle.dumps(_Evil()))).load()


def test_restricted_unpickler_loads_numpy_and_remaps_legacy_paths() -> None:
    arr = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert np.array_equal(RestrictedUnpickler(io.BytesIO(pickle.dumps(arr))).load(), arr)
    u = RestrictedUnpickler(io.BytesIO(b""))
    # module paths written by the 2015 Python-2 pandas that produced LSWMD.pkl
    assert u.find_class("__builtin__", "slice") is slice
    assert u.find_class("pandas.indexes.base", "_new_Index") is pd.core.indexes.base._new_Index
    assert u.find_class("pandas.core.frame", "DataFrame") is pd.DataFrame
    with pytest.raises(pickle.UnpicklingError):
        u.find_class("builtins", "eval")


def test_flatten_labels() -> None:
    assert _flatten(np.array([["Center"]])) == "Center"
    assert _flatten(np.array([])) is None


def test_splits_are_lot_grouped_and_stratified() -> None:
    rng = np.random.default_rng(0)
    lots = np.repeat([f"lot{i}" for i in range(600)], 25)
    labels = rng.choice(["a", "b", "none"], size=len(lots), p=[0.1, 0.1, 0.8])
    split = assign_splits(labels, lots, seed=1)
    for s1, s2 in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(lots[split == s1]) & set(lots[split == s2])
    shares = {s: np.mean(split == s) for s in ("train", "val", "test")}
    assert shares["train"] == pytest.approx(0.70, abs=0.03)
    assert shares["test"] == pytest.approx(0.15, abs=0.03)
    for c in ("a", "b"):  # each class keeps roughly its share in test
        assert np.mean(labels[split == "test"] == c) == pytest.approx(
            np.mean(labels == c), abs=0.03
        )


def test_features_shape_and_determinism() -> None:
    assert len(feature_names()) == N_FEATURES
    f1, f2 = extract(center()), extract(center())
    assert f1.shape == (N_FEATURES,)
    assert np.array_equal(f1, f2)
    assert np.isfinite(extract(blank())).all()  # no fails -> no NaN


def test_features_separate_basic_patterns() -> None:
    names = feature_names()
    c, r, s = extract(center()), extract(edge_ring()), extract(scratch())
    ring0, ring4 = names.index("ring_0"), names.index("ring_4")
    assert c[ring0] > 0.9 and c[ring4] < 0.05
    assert r[ring4] > 0.5 and r[ring0] < 0.05
    ecc = names.index("largest_eccentricity")
    assert s[ecc] > 0.95  # a line is maximally elongated


def test_macro_f1_matches_sklearn() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 9, 2000)
    p = np.where(rng.random(2000) < 0.7, y, rng.integers(0, 9, 2000))
    assert macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro", labels=range(9)))
    assert confusion(y, p).sum() == 2000


def test_evaluate_and_bootstrap() -> None:
    rng = np.random.default_rng(4)
    y = rng.integers(0, 9, 3000)
    p = np.where(rng.random(3000) < 0.8, y, rng.integers(0, 9, 3000))
    groups = np.repeat(np.arange(300), 10)
    out = evaluate(y, p, groups=groups)
    lo, hi = out["macro_f1_ci95_lot_bootstrap"]
    assert lo <= out["macro_f1"] <= hi
    assert sum(v["support"] for v in out["per_class"].values()) == 3000
    lo2, hi2 = bootstrap_macro_f1(y, p, groups, n_boot=200, seed=1)
    assert 0 < lo2 < hi2 < 1
