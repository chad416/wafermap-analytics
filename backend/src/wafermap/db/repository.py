"""Query layer. Endpoints call these functions; they never build SQL themselves."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import Integer, Select, and_, cast, exists, func, select, update
from sqlalchemy.orm import Session, selectinload, undefer

from wafermap.constants import CLASSES
from wafermap.db.models import Lot, Prediction, Wafer

LabelSource = Literal["true", "predicted"]
LotSort = Literal["lot_number", "mean_yield", "wafer_count", "name"]
WaferSort = Literal["wafer_index", "yield_pct", "id"]


@dataclass(frozen=True)
class PageResult:
    items: list[Any]
    total: int


def refresh_lot_stats(s: Session, lot_ids: list[int] | None = None) -> None:
    count_q = select(func.count(Wafer.id)).where(Wafer.lot_id == Lot.id).scalar_subquery()
    mean_q = select(func.avg(Wafer.yield_pct)).where(Wafer.lot_id == Lot.id).scalar_subquery()
    stmt = update(Lot).values(wafer_count=count_q, mean_yield=mean_q)
    if lot_ids is not None:
        stmt = stmt.where(Lot.id.in_(lot_ids))
    s.execute(stmt)


def _pattern_filter(label: str, source: LabelSource, model: str) -> Any:
    """SQL condition on ``Wafer`` matching a true or predicted pattern label."""
    if source == "true":
        return Wafer.true_label == label
    return exists().where(and_(Prediction.wafer_id == Wafer.id, Prediction.model_name == model,
                               Prediction.label == label))


def list_lots(
    s: Session,
    *,
    page: int,
    page_size: int,
    sort: LotSort = "lot_number",
    descending: bool = False,
    q: str | None = None,
    pattern: str | None = None,
    source: LabelSource = "predicted",
    model: str = "cnn",
    min_yield: float | None = None,
    max_yield: float | None = None,
) -> PageResult:
    stmt: Select[Any] = select(Lot)
    if q:
        stmt = stmt.where(Lot.name.ilike(f"%{q}%"))
    if min_yield is not None:
        stmt = stmt.where(Lot.mean_yield >= min_yield)
    if max_yield is not None:
        stmt = stmt.where(Lot.mean_yield <= max_yield)
    if pattern:
        stmt = stmt.where(exists().where(and_(Wafer.lot_id == Lot.id, _pattern_filter(pattern, source, model))))
    total = s.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    col = getattr(Lot, sort)
    stmt = stmt.order_by(col.desc() if descending else col.asc(), Lot.id).offset((page - 1) * page_size).limit(page_size)
    lots = list(s.scalars(stmt))
    return PageResult(items=lots, total=total)


def lot_pattern_counts(
    s: Session, lot_ids: list[int], source: LabelSource, model: str
) -> dict[int, Counter[str]]:
    if not lot_ids:
        return {}
    if source == "true":
        stmt = (select(Wafer.lot_id, Wafer.true_label, func.count()).where(Wafer.lot_id.in_(lot_ids),
                Wafer.true_label.is_not(None)).group_by(Wafer.lot_id, Wafer.true_label))
    else:
        stmt = (select(Wafer.lot_id, Prediction.label, func.count()).join(Prediction, Prediction.wafer_id == Wafer.id)
                .where(Wafer.lot_id.in_(lot_ids), Prediction.model_name == model)
                .group_by(Wafer.lot_id, Prediction.label))
    out: dict[int, Counter[str]] = defaultdict(Counter)
    for lot_id, label, n in s.execute(stmt):
        out[lot_id][label] = n
    return out


def get_lot(s: Session, name: str) -> Lot | None:
    return s.scalar(select(Lot).where(Lot.name == name).options(
        selectinload(Lot.wafers).selectinload(Wafer.predictions)))


def list_wafers(
    s: Session,
    *,
    page: int,
    page_size: int,
    lot: str | None = None,
    pattern: str | None = None,
    source: LabelSource = "predicted",
    model: str = "cnn",
    labelled_only: bool = False,
    min_yield: float | None = None,
    max_yield: float | None = None,
    sort: WaferSort = "id",
    descending: bool = False,
) -> PageResult:
    stmt: Select[Any] = select(Wafer).join(Lot)
    if lot:
        stmt = stmt.where(Lot.name == lot)
    if pattern:
        stmt = stmt.where(_pattern_filter(pattern, source, model))
    if labelled_only:
        stmt = stmt.where(Wafer.true_label.is_not(None))
    if min_yield is not None:
        stmt = stmt.where(Wafer.yield_pct >= min_yield)
    if max_yield is not None:
        stmt = stmt.where(Wafer.yield_pct <= max_yield)
    total = s.scalar(select(func.count()).select_from(stmt.with_only_columns(Wafer.id).subquery())) or 0
    col = getattr(Wafer, sort)
    stmt = (stmt.options(selectinload(Wafer.lot), selectinload(Wafer.predictions))
            .order_by(col.desc() if descending else col.asc(), Wafer.id)
            .offset((page - 1) * page_size).limit(page_size))
    return PageResult(items=list(s.scalars(stmt)), total=total)


def get_wafer(s: Session, wafer_id: int, *, with_map: bool = False) -> Wafer | None:
    stmt = select(Wafer).where(Wafer.id == wafer_id).options(selectinload(Wafer.lot), selectinload(Wafer.predictions))
    if with_map:
        stmt = stmt.options(undefer(Wafer.map_zlib))
    return s.scalar(stmt)


def yield_trend(
    s: Session, *, bucket_lots: int, pattern: str | None, source: LabelSource, model: str
) -> list[dict[str, Any]]:
    """Mean wafer yield per bucket of ``bucket_lots`` consecutive lot numbers."""
    bucket = cast((Lot.lot_number - 1) / bucket_lots, Integer)
    stmt = (select(bucket.label("bucket"), func.min(Lot.lot_number), func.max(Lot.lot_number),
                   func.count(Wafer.id), func.avg(Wafer.yield_pct), func.min(Wafer.yield_pct),
                   func.count(func.distinct(Lot.id)))
            .join(Lot, Wafer.lot_id == Lot.id).where(Lot.source == "wm811k"))
    if pattern:
        stmt = stmt.where(_pattern_filter(pattern, source, model))
    stmt = stmt.group_by(bucket).order_by(bucket)
    return [
        {"bucket": int(b), "lot_number_from": int(lo), "lot_number_to": int(hi), "wafers": int(n),
         "lots": int(nl), "mean_yield": round(float(avg), 3), "min_yield": round(float(mn), 3)}
        for b, lo, hi, n, avg, mn, nl in s.execute(stmt)
    ]


def pattern_distribution(
    s: Session, *, source: LabelSource, model: str, lot: str | None = None
) -> list[dict[str, Any]]:
    if source == "true":
        label_col = Wafer.true_label
        stmt = select(label_col, func.count(Wafer.id), func.avg(Wafer.yield_pct)).where(label_col.is_not(None))
    else:
        label_col = Prediction.label
        stmt = (select(label_col, func.count(Wafer.id), func.avg(Wafer.yield_pct))
                .join(Prediction, Prediction.wafer_id == Wafer.id).where(Prediction.model_name == model))
    if lot:
        stmt = stmt.join(Lot, Wafer.lot_id == Lot.id).where(Lot.name == lot)
    rows = {label: (int(n), float(avg)) for label, n, avg in s.execute(stmt.group_by(label_col))}
    total = sum(n for n, _ in rows.values()) or 1
    return [
        {"pattern": c, "count": rows.get(c, (0, 0.0))[0], "share_pct": round(100 * rows.get(c, (0, 0.0))[0] / total, 3),
         "mean_yield": round(rows[c][1], 3) if c in rows else None}
        for c in CLASSES
    ]


def summary(s: Session, model: str) -> dict[str, Any]:
    n_lots = s.scalar(select(func.count(Lot.id))) or 0
    n_wafers, mean_yield = s.execute(select(func.count(Wafer.id), func.avg(Wafer.yield_pct))).one()
    labelled = s.scalar(select(func.count(Wafer.id)).where(Wafer.true_label.is_not(None))) or 0
    predicted = s.scalar(select(func.count(Prediction.id)).where(Prediction.model_name == model)) or 0
    defect = s.scalar(select(func.count(Prediction.id)).where(Prediction.model_name == model,
                                                             Prediction.label != "none")) or 0
    uploads = s.scalar(select(func.count(Wafer.id)).where(Wafer.source == "upload")) or 0
    return {
        "lots": n_lots,
        "wafers": int(n_wafers or 0),
        "labelled_wafers": labelled,
        "predicted_wafers": predicted,
        "uploaded_wafers": uploads,
        "mean_yield": round(float(mean_yield), 3) if mean_yield is not None else None,
        "predicted_defect_share_pct": round(100 * defect / predicted, 3) if predicted else None,
    }
