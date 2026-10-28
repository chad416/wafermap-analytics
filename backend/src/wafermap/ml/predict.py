"""Runtime inference. No PyTorch: the CNN runs on ONNX Runtime, the baseline on LightGBM."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from wafermap import wafer
from wafermap.constants import CLASSES
from wafermap.wafer import WaferArray

log = logging.getLogger(__name__)
FloatArray = npt.NDArray[np.float32]


class Classifier(Protocol):
    name: str
    version: str
    meta: dict[str, Any]

    def predict_proba(self, maps: list[WaferArray]) -> FloatArray: ...


def softmax(z: FloatArray) -> FloatArray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


@dataclass
class CnnOnnxClassifier:
    path: Path
    meta: dict[str, Any]
    name: str = "cnn"
    version: str = field(init=False)

    def __post_init__(self) -> None:
        self.version = str(self.meta["version"])

    @cached_property
    def _session(self) -> Any:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        return ort.InferenceSession(str(self.path), sess_options=opts, providers=["CPUExecutionProvider"])

    def predict_proba(self, maps: list[WaferArray]) -> FloatArray:
        x = np.stack([wafer.canonical_u8(m) for m in maps]).astype(np.float32) / 255.0
        return softmax(self._session.run(None, {"x": x})[0])


@dataclass
class LightGBMClassifier:
    path: Path
    meta: dict[str, Any]
    name: str = "baseline"
    version: str = field(init=False)

    def __post_init__(self) -> None:
        self.version = str(self.meta["version"])

    @cached_property
    def _booster(self) -> Any:
        import lightgbm as lgb

        return lgb.Booster(model_file=str(self.path))

    def predict_proba(self, maps: list[WaferArray]) -> FloatArray:
        from wafermap.features.engineered import extract

        x = np.stack([extract(m) for m in maps])
        return np.asarray(self._booster.predict(x), dtype=np.float32)


class ModelRegistry:
    """Discovers trained models under ``model_dir`` (``<name>/meta.json`` + weights)."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._models: dict[str, Classifier] = {}
        for name, weights, cls in (("cnn", "model.onnx", CnnOnnxClassifier),
                                   ("baseline", "model.txt", LightGBMClassifier)):
            meta_path, w_path = model_dir / name / "meta.json", model_dir / name / weights
            if meta_path.exists() and w_path.exists():
                meta = json.loads(meta_path.read_text())
                if tuple(meta["classes"]) != CLASSES:
                    raise ValueError(f"{name}: class list in meta.json does not match CLASSES")
                self._models[name] = cls(path=w_path, meta=meta)
            else:
                log.warning("model %s not found in %s", name, model_dir)

    def names(self) -> list[str]:
        return list(self._models)

    def get(self, name: str) -> Classifier:
        try:
            return self._models[name]
        except KeyError:
            raise KeyError(f"model '{name}' is not available (have: {self.names()})") from None


def top(proba: FloatArray) -> list[tuple[str, float, dict[str, float]]]:
    """(label, confidence, {class: p}) per row."""
    out = []
    for row in proba:
        i = int(row.argmax())
        out.append((CLASSES[i], float(row[i]), {c: round(float(p), 5) for c, p in zip(CLASSES, row, strict=True)}))
    return out
