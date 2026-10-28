"""Live classification of a new wafer map (JSON grid or file upload), optionally persisted."""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from wafermap import wafer
from wafermap.api.deps import RegistryDep, SessionDep, SettingsDep
from wafermap.api.schemas import ModelName, PredictRequest, PredictResponse, WaferStatsOut
from wafermap.db.models import Lot, Prediction, Wafer
from wafermap.db.repository import refresh_lot_stats
from wafermap.ml.predict import ModelRegistry, top
from wafermap.wafer import InvalidWaferMapError, WaferArray

router = APIRouter(prefix="/predict", tags=["prediction"])
UPLOAD_LOT = "uploads"
ERRORS: dict[int | str, dict[str, Any]] = {
    413: {"description": "File too large"},
    422: {"description": "Not a valid wafer map"},
    503: {"description": "Model not available"},
}


def _classify(
    m: WaferArray, model: str, registry: ModelRegistry
) -> tuple[str, str, str, float, dict[str, float], float]:
    try:
        clf = registry.get(model)
    except KeyError as exc:
        raise HTTPException(503, str(exc)) from exc
    t0 = time.perf_counter()
    label, conf, probs = top(clf.predict_proba([m]))[0]
    return clf.name, clf.version, label, conf, probs, (time.perf_counter() - t0) * 1000


def _save(
    session: Session, m: WaferArray, preds: list[tuple[str, str, str, float, dict[str, float]]]
) -> int:
    lot = session.scalar(select(Lot).where(Lot.name == UPLOAD_LOT))
    if lot is None:
        lot = Lot(name=UPLOAD_LOT, lot_number=0, source="upload")
        session.add(lot)
        session.flush()
    next_index = (
        session.scalar(select(func.max(Wafer.wafer_index)).where(Wafer.lot_id == lot.id)) or 0
    ) + 1
    st = wafer.stats(m)
    w = Wafer(
        lot_id=lot.id,
        wafer_index=next_index,
        rows=m.shape[0],
        cols=m.shape[1],
        n_die=st.n_die,
        n_pass=st.n_pass,
        n_fail=st.n_fail,
        yield_pct=round(st.yield_pct, 3),
        map_zlib=wafer.encode(m),
        source="upload",
    )
    w.predictions = [
        Prediction(model_name=n, model_version=v, label=lab, confidence=c, probabilities=p)
        for n, v, lab, c, p in preds
    ]
    session.add(w)
    session.flush()
    refresh_lot_stats(session, [lot.id])
    session.commit()
    return w.id


def _respond(
    m: WaferArray,
    model: str,
    save: bool,
    session: Session,
    registry: ModelRegistry,
    default_model: str,
    echo_grid: bool,
) -> PredictResponse:
    name, version, label, conf, probs, ms = _classify(m, model, registry)
    wafer_id = None
    if save:
        preds = [(name, version, label, conf, probs)]
        if model != default_model and default_model in registry.names():
            preds.append(_classify(m, default_model, registry)[:5])
        wafer_id = _save(session, m, preds)
    st = wafer.stats(m)
    return PredictResponse(
        model_name=name,
        model_version=version,
        label=label,
        confidence=round(conf, 5),
        probabilities=probs,
        stats=WaferStatsOut(
            rows=m.shape[0],
            cols=m.shape[1],
            n_die=st.n_die,
            n_fail=st.n_fail,
            yield_pct=round(st.yield_pct, 3),
        ),
        latency_ms=round(ms, 2),
        wafer_id=wafer_id,
        grid=m.tolist() if echo_grid else None,
    )


@router.post(
    "", response_model=PredictResponse, responses=ERRORS, summary="Classify a JSON die grid"
)
def predict_grid(
    body: PredictRequest,
    session: SessionDep,
    settings: SettingsDep,
    registry: RegistryDep,
    model: ModelName | None = None,
    save: Annotated[bool, Query(description="Persist the wafer under lot 'uploads'")] = False,
) -> PredictResponse:
    if len({len(r) for r in body.grid}) != 1:
        raise HTTPException(422, "all grid rows must have the same length")
    try:
        m = wafer.validate(body.grid)
    except InvalidWaferMapError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _respond(
        m, model or settings.default_model, save, session, registry, settings.default_model, False
    )


@router.post(
    "/upload",
    response_model=PredictResponse,
    responses=ERRORS,
    summary="Classify an uploaded map file (.npy, .json, .csv, .txt)",
)
async def predict_upload(
    session: SessionDep,
    settings: SettingsDep,
    registry: RegistryDep,
    file: Annotated[
        UploadFile, File(description="Wafer map: .npy, .json (2-D list) or CSV/TXT grid")
    ],
    model: ModelName | None = None,
    save: bool = False,
) -> PredictResponse:
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, f"file exceeds {settings.max_upload_bytes} bytes")
    try:
        m = wafer.parse_upload(data, file.filename or "")
    except InvalidWaferMapError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _respond(
        m, model or settings.default_model, save, session, registry, settings.default_model, True
    )
