from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from wafermap.api.deps import PaginationDep, SessionDep, SettingsDep
from wafermap.api.schemas import LabelSource, LotDetail, LotOut, Page, PatternName
from wafermap.api.serializers import lot_out, wafer_summary
from wafermap.db import repository as repo

router = APIRouter(prefix="/lots", tags=["lots"])


@router.get("", response_model=Page[LotOut], summary="List lots")
def list_lots(
    session: SessionDep,
    settings: SettingsDep,
    pagination: PaginationDep,
    q: Annotated[str | None, Query(max_length=64, description="Substring of the lot name")] = None,
    pattern: Annotated[PatternName | None, Query(description="Only lots containing this pattern")] = None,
    source: LabelSource = "predicted",
    min_yield: Annotated[float | None, Query(ge=0, le=100)] = None,
    max_yield: Annotated[float | None, Query(ge=0, le=100)] = None,
    sort: repo.LotSort = "lot_number",
    descending: bool = False,
) -> Page[LotOut]:
    model = settings.default_model
    res = repo.list_lots(session, page=pagination.page, page_size=pagination.page_size, sort=sort,
                         descending=descending, q=q, pattern=pattern, source=source, model=model,
                         min_yield=min_yield, max_yield=max_yield)
    counts = repo.lot_pattern_counts(session, [lot.id for lot in res.items], source, model)
    return Page(items=[lot_out(lot, counts.get(lot.id, {})) for lot in res.items],  # type: ignore[arg-type]
                total=res.total, page=pagination.page, page_size=pagination.page_size,
                pages=pagination.pages(res.total))


@router.get("/{lot_name}", response_model=LotDetail, summary="Lot with all its wafers",
            responses={404: {"description": "Lot not found"}})
def get_lot(lot_name: str, session: SessionDep, settings: SettingsDep, source: LabelSource = "predicted") -> LotDetail:
    lot = repo.get_lot(session, lot_name)
    if lot is None:
        raise HTTPException(404, f"lot '{lot_name}' not found")
    model = settings.default_model
    counts = repo.lot_pattern_counts(session, [lot.id], source, model).get(lot.id, {})
    base = lot_out(lot, counts)  # type: ignore[arg-type]
    return LotDetail(**base.model_dump(), wafers=[wafer_summary(w, model) for w in lot.wafers])
