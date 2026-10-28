from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from wafermap import wafer
from wafermap.api.deps import PaginationDep, SessionDep, SettingsDep
from wafermap.api.schemas import (
    LabelSource,
    Page,
    PatternName,
    WaferDetail,
    WaferMapOut,
    WaferSummary,
)
from wafermap.api.serializers import wafer_detail, wafer_summary
from wafermap.db import repository as repo

router = APIRouter(prefix="/wafers", tags=["wafers"])
NOT_FOUND: dict[int | str, dict[str, Any]] = {404: {"description": "Wafer not found"}}


@router.get("", response_model=Page[WaferSummary], summary="List / filter wafers")
def list_wafers(
    session: SessionDep,
    settings: SettingsDep,
    pagination: PaginationDep,
    lot: Annotated[str | None, Query(max_length=64)] = None,
    pattern: PatternName | None = None,
    source: LabelSource = "predicted",
    labelled_only: bool = False,
    min_yield: Annotated[float | None, Query(ge=0, le=100)] = None,
    max_yield: Annotated[float | None, Query(ge=0, le=100)] = None,
    sort: repo.WaferSort = "id",
    descending: bool = False,
) -> Page[WaferSummary]:
    model = settings.default_model
    res = repo.list_wafers(
        session,
        page=pagination.page,
        page_size=pagination.page_size,
        lot=lot,
        pattern=pattern,
        source=source,
        model=model,
        labelled_only=labelled_only,
        min_yield=min_yield,
        max_yield=max_yield,
        sort=sort,
        descending=descending,
    )
    return Page(
        items=[wafer_summary(w, model) for w in res.items],
        total=res.total,
        page=pagination.page,
        page_size=pagination.page_size,
        pages=pagination.pages(res.total),
    )


@router.get("/{wafer_id}", response_model=WaferDetail, summary="Wafer detail", responses=NOT_FOUND)
def get_wafer(wafer_id: int, session: SessionDep, settings: SettingsDep) -> WaferDetail:
    w = repo.get_wafer(session, wafer_id)
    if w is None:
        raise HTTPException(404, f"wafer {wafer_id} not found")
    return wafer_detail(w, settings.default_model)


@router.get(
    "/{wafer_id}/map",
    response_model=WaferMapOut,
    summary="Die-level wafer map",
    responses=NOT_FOUND,
)
def get_wafer_map(wafer_id: int, session: SessionDep) -> WaferMapOut:
    w = repo.get_wafer(session, wafer_id, with_map=True)
    if w is None:
        raise HTTPException(404, f"wafer {wafer_id} not found")
    grid = wafer.decode(w.map_zlib, w.rows, w.cols)
    return WaferMapOut(id=w.id, rows=w.rows, cols=w.cols, grid=grid.tolist())
