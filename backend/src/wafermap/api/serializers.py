"""ORM -> response-model conversion."""

from __future__ import annotations

from collections import Counter

from wafermap.api.schemas import LotOut, PredictionOut, WaferDetail, WaferSummary
from wafermap.db.models import Lot, Prediction, Wafer


def prediction_out(p: Prediction) -> PredictionOut:
    return PredictionOut.model_validate(p)


def _pick(w: Wafer, model: str) -> Prediction | None:
    return next((p for p in w.predictions if p.model_name == model), None)


def wafer_summary(w: Wafer, model: str) -> WaferSummary:
    p = _pick(w, model)
    return WaferSummary(
        id=w.id,
        lot_name=w.lot.name,
        wafer_index=w.wafer_index,
        rows=w.rows,
        cols=w.cols,
        n_die=w.n_die,
        n_fail=w.n_fail,
        yield_pct=w.yield_pct,
        true_label=w.true_label,
        split=w.split,
        source=w.source,
        prediction=prediction_out(p) if p else None,
    )


def wafer_detail(w: Wafer, model: str) -> WaferDetail:
    return WaferDetail(
        **wafer_summary(w, model).model_dump(),
        predictions=[prediction_out(p) for p in w.predictions],
    )


def lot_out(lot: Lot, counts: Counter[str]) -> LotOut:
    defects = [(n, c) for c, n in counts.items() if c != "none"]
    top_defect = max(defects)[1] if defects else None
    return LotOut(
        name=lot.name,
        lot_number=lot.lot_number,
        source=lot.source,
        wafer_count=lot.wafer_count,
        mean_yield=round(lot.mean_yield, 3) if lot.mean_yield is not None else None,
        pattern_counts=dict(counts),
        top_defect=top_defect,
    )
