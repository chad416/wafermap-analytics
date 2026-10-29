"""CNN building blocks: shapes, ONNX export parity, augmentation and sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # training extra; the API image has no torch

from wafermap.constants import CANONICAL_SIZE, CLASSES  # noqa: E402
from wafermap.ml.cnn import WaferCNN  # noqa: E402
from wafermap.ml.train_cnn import (  # noqa: E402
    dihedral,
    export_onnx,
    onnx_parity,
    sqrt_sampling_weights,
)


def test_forward_shape_and_param_count() -> None:
    model = WaferCNN().eval()
    out = model(torch.zeros(3, 2, CANONICAL_SIZE, CANONICAL_SIZE))
    assert out.shape == (3, len(CLASSES))
    assert sum(p.numel() for p in model.parameters()) == 295_017


def test_onnx_export_matches_pytorch(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = WaferCNN().eval()
    path = tmp_path / "m.onnx"
    export_onnx(model, path)
    x = np.random.default_rng(0).integers(
        0, 256, (16, 2, CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.uint8
    )
    assert onnx_parity(path, model, x) < 1e-4


def test_dihedral_preserves_content() -> None:
    x = torch.rand(64, 2, 8, 8)
    y = dihedral(x, torch.Generator().manual_seed(1))
    assert y.shape == x.shape
    # each output is a rotation/flip of its input: same multiset of pixel values per sample
    assert torch.allclose(y.flatten(1).sort(dim=1).values, x.flatten(1).sort(dim=1).values)
    assert not torch.equal(x, y)  # with 64 samples some transform is non-identity


def test_sqrt_sampling_gives_sqrt_class_mass() -> None:
    y = np.array([0] * 100 + [1] * 25 + [2] * 4)
    p = sqrt_sampling_weights(y)
    assert p.sum() == pytest.approx(1.0)
    mass = np.array([p[y == c].sum() for c in range(3)])
    np.testing.assert_allclose(
        mass / mass.sum(), np.sqrt([100, 25, 4]) / np.sqrt([100, 25, 4]).sum()
    )
