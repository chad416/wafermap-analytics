from __future__ import annotations

import io
import json

import numpy as np
import pytest

from synthetic import center, edge_ring
from wafermap import wafer
from wafermap.wafer import InvalidWaferMap


def test_encode_decode_roundtrip() -> None:
    m = edge_ring(37)
    assert np.array_equal(wafer.decode(wafer.encode(m), *m.shape), m)


def test_decode_rejects_wrong_shape() -> None:
    m = center()
    with pytest.raises(InvalidWaferMap):
        wafer.decode(wafer.encode(m), m.shape[0] + 1, m.shape[1])


def test_stats_and_yield() -> None:
    m = np.array([[0, 1, 1], [1, 2, 1], [0, 1, 2]] * 20, dtype=np.uint8)
    st = wafer.stats(m)
    assert (st.n_die, st.n_fail) == (140, 40)
    assert st.yield_pct == pytest.approx(100 * 100 / 140)


@pytest.mark.parametrize(
    ("arr", "msg"),
    [
        (np.ones(50), "2-D"),
        (np.full((20, 20), 3), "values must be"),
        (np.ones((2, 200)), "too small"),
        (np.ones((600, 10)), "too large"),
        (np.ones((5, 5)), "at least"),
        (np.full((20, 20), np.nan), "NaN"),
    ],
)
def test_validate_rejects(arr: np.ndarray, msg: str) -> None:
    with pytest.raises(InvalidWaferMap, match=msg):
        wafer.validate(arr)


@pytest.mark.parametrize("size", [8, 25, 64, 97, 300])
def test_area_resize_preserves_fractions(size: int) -> None:
    rng = np.random.default_rng(size)
    m = rng.integers(0, 3, (size, size + 7)).astype(np.uint8)
    die_c, fail_c = wafer.canonical(m)
    assert die_c.shape == fail_c.shape == (64, 64)
    assert die_c.mean() == pytest.approx((m > 0).mean(), abs=1e-5)
    assert fail_c.mean() == pytest.approx((m == 2).mean(), abs=1e-5)


def test_canonical_u8_matches_float() -> None:
    m = center()
    assert np.abs(wafer.canonical_u8(m) / 255.0 - wafer.canonical(m)).max() <= 0.5 / 255 + 1e-6


def test_parse_upload_formats() -> None:
    m = center(30)
    buf = io.BytesIO()
    np.save(buf, m)
    assert np.array_equal(wafer.parse_upload(buf.getvalue(), "w.npy"), m)
    assert np.array_equal(wafer.parse_upload(json.dumps(m.tolist()).encode(), "w.json"), m)
    assert np.array_equal(wafer.parse_upload(json.dumps({"grid": m.tolist()}).encode(), "w.JSON"), m)
    csv = "\n".join(",".join(map(str, r)) for r in m).encode()
    assert np.array_equal(wafer.parse_upload(csv, "w.csv"), m)
    txt = "\n".join(" ".join(map(str, r)) for r in m).encode()
    assert np.array_equal(wafer.parse_upload(txt, "w.txt"), m)


def test_parse_upload_rejects_pickled_npy() -> None:
    buf = io.BytesIO()
    np.save(buf, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(InvalidWaferMap):
        wafer.parse_upload(buf.getvalue(), "evil.npy")


@pytest.mark.parametrize(
    ("payload", "name"),
    [(b"1,2\n1", "w.csv"), (b"not json", "w.json"), (b"\x00\x01", "w.bin"), (b"a,b\nc,d", "w.csv")],
)
def test_parse_upload_errors(payload: bytes, name: str) -> None:
    with pytest.raises(InvalidWaferMap):
        wafer.parse_upload(payload, name)
