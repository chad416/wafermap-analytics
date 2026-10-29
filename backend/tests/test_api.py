from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import HAVE_MODELS
from synthetic import center, edge_ring

needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="trained models not present in models/")


def test_health(client: TestClient) -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["database"] is True


def test_openapi_lists_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for p in (
        "/api/v1/lots",
        "/api/v1/lots/{lot_name}",
        "/api/v1/wafers",
        "/api/v1/wafers/{wafer_id}/map",
        "/api/v1/predict",
        "/api/v1/predict/upload",
        "/api/v1/stats/yield-trend",
        "/api/v1/stats/patterns",
        "/api/v1/stats/summary",
        "/api/v1/models",
    ):
        assert p in paths


def test_lots_pagination(client: TestClient) -> None:
    r = client.get("/api/v1/lots", params={"page_size": 2})
    body = r.json()
    assert r.status_code == 200
    assert body["total"] >= 3 and body["pages"] >= 2 and len(body["items"]) == 2
    page2 = client.get("/api/v1/lots", params={"page_size": 2, "page": 2}).json()
    assert {x["name"] for x in body["items"]}.isdisjoint({x["name"] for x in page2["items"]})


def test_lots_filter_and_sort(client: TestClient) -> None:
    lots = client.get("/api/v1/lots", params={"pattern": "Scratch", "source": "true"}).json()[
        "items"
    ]
    assert {x["name"] for x in lots} == {"lot1", "lot2"}  # lot3 is unlabelled
    desc = client.get("/api/v1/lots", params={"sort": "lot_number", "descending": True}).json()[
        "items"
    ]
    numbers = [x["lot_number"] for x in desc]
    assert numbers == sorted(numbers, reverse=True)
    lot1 = next(x for x in lots if x["name"] == "lot1")
    assert lot1["wafer_count"] == 4 and lot1["top_defect"] is not None


@pytest.mark.parametrize(
    "params",
    [
        {"page_size": 500},
        {"page": 0},
        {"pattern": "Banana"},
        {"min_yield": 120},
        {"sort": "drop table"},
    ],
)
def test_lots_validation(client: TestClient, params: dict[str, object]) -> None:
    assert client.get("/api/v1/lots", params=params).status_code == 422


def test_lot_detail_and_404(client: TestClient) -> None:
    r = client.get("/api/v1/lots/lot2")
    assert r.status_code == 200
    body = r.json()
    assert [w["wafer_index"] for w in body["wafers"]] == [1, 2, 3, 4]
    assert body["wafers"][0]["prediction"]["model_name"] == "cnn"
    assert client.get("/api/v1/lots/nope").status_code == 404


def test_wafers_list_filters(client: TestClient) -> None:
    r = client.get("/api/v1/wafers", params={"lot": "lot1", "pattern": "Center"}).json()
    assert r["total"] == 1 and r["items"][0]["true_label"] == "Center"
    r = client.get("/api/v1/wafers", params={"labelled_only": True, "page_size": 100}).json()
    assert r["total"] == 8
    r = client.get("/api/v1/wafers", params={"sort": "yield_pct", "page_size": 100}).json()
    ys = [w["yield_pct"] for w in r["items"]]
    assert ys == sorted(ys)


def test_wafer_map_roundtrip(client: TestClient) -> None:
    wid = client.get("/api/v1/wafers", params={"lot": "lot1", "pattern": "Center"}).json()["items"][
        0
    ]["id"]
    m = client.get(f"/api/v1/wafers/{wid}/map").json()
    assert np.array_equal(np.asarray(m["grid"]), center())
    assert client.get(f"/api/v1/wafers/{wid}").json()["predictions"][0]["label"] == "Center"
    assert client.get("/api/v1/wafers/999999/map").status_code == 404


def test_stats(client: TestClient) -> None:
    s = client.get("/api/v1/stats/summary").json()
    assert s["lots"] >= 3 and s["labelled_wafers"] == 8
    trend = client.get("/api/v1/stats/yield-trend", params={"bucket_lots": 2}).json()
    assert [t["lots"] for t in trend] == [2, 1]
    dist = {
        d["pattern"]: d
        for d in client.get("/api/v1/stats/patterns", params={"source": "true"}).json()
    }
    assert dist["Center"]["count"] == 2 and dist["Donut"]["count"] == 0
    assert sum(d["share_pct"] for d in dist.values()) == pytest.approx(100, abs=0.01)
    by_lot = client.get("/api/v1/stats/patterns", params={"lot": "lot3"}).json()
    assert sum(d["count"] for d in by_lot) == 4


def test_predict_rejects_bad_input(client: TestClient) -> None:
    assert client.post("/api/v1/predict", json={"grid": [[1, 2], [1]]}).status_code == 422
    assert client.post("/api/v1/predict", json={"grid": [[5] * 20] * 20}).status_code == 422
    assert client.post("/api/v1/predict", json={"grid": "nope"}).status_code == 422
    r = client.post(
        "/api/v1/predict/upload", files={"file": ("w.exe", b"MZ", "application/octet-stream")}
    )
    assert r.status_code == 422


def test_upload_too_large(client: TestClient) -> None:
    big = b"1," * (2 * 1024 * 1024)
    r = client.post("/api/v1/predict/upload", files={"file": ("big.csv", big, "text/csv")})
    assert r.status_code == 413


@needs_models
@pytest.mark.parametrize("model", ["cnn", "baseline"])
def test_predict_grid(client: TestClient, model: str) -> None:
    r = client.post("/api/v1/predict", params={"model": model}, json={"grid": edge_ring().tolist()})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_name"] == model
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["label"] in body["probabilities"]
    assert body["wafer_id"] is None


@needs_models
def test_upload_and_save(client: TestClient) -> None:
    buf = io.BytesIO()
    np.save(buf, center())
    r = client.post(
        "/api/v1/predict/upload",
        params={"save": True},
        files={"file": ("w.npy", buf.getvalue(), "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["wafer_id"] is not None and body["grid"] == center().tolist()
    lot = client.get("/api/v1/lots/uploads").json()
    assert lot["source"] == "upload" and lot["wafer_count"] >= 1
    detail = client.get(f"/api/v1/wafers/{body['wafer_id']}").json()
    assert detail["source"] == "upload" and detail["prediction"]["label"] == body["label"]
    # uploads never pollute the production yield trend
    trend = client.get("/api/v1/stats/yield-trend", params={"bucket_lots": 2}).json()
    assert sum(t["wafers"] for t in trend) == 12


@needs_models
def test_models_endpoint(client: TestClient) -> None:
    names = {m["name"] for m in client.get("/api/v1/models").json()}
    assert names == {"cnn", "baseline"}


@needs_models
def test_models_are_loaded_at_startup(client: TestClient) -> None:
    # warm_up() in the lifespan: the first request must not pay for model loading
    registry = client.app.state.registry  # type: ignore[attr-defined]
    assert "_session" in vars(registry.get("cnn"))
    assert "_booster" in vars(registry.get("baseline"))
