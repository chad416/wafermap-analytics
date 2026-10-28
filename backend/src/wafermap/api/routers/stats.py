from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from wafermap.api.deps import RegistryDep, SessionDep, SettingsDep
from wafermap.api.schemas import (
    LabelSource,
    ModelInfo,
    PatternCount,
    PatternName,
    Summary,
    YieldTrendPoint,
)
from wafermap.db import repository as repo

router = APIRouter(tags=["analytics"])


@router.get("/stats/summary", response_model=Summary, summary="Fleet-level KPIs")
def summary(session: SessionDep, settings: SettingsDep) -> Summary:
    return Summary(
        **repo.summary(session, settings.default_model), default_model=settings.default_model
    )


@router.get(
    "/stats/yield-trend",
    response_model=list[YieldTrendPoint],
    summary="Mean yield along lot sequence",
)
def yield_trend(
    session: SessionDep,
    settings: SettingsDep,
    bucket_lots: Annotated[int, Query(ge=1, le=10_000, description="Lot numbers per bucket")] = 500,
    pattern: PatternName | None = None,
    source: LabelSource = "predicted",
) -> list[YieldTrendPoint]:
    rows = repo.yield_trend(
        session,
        bucket_lots=bucket_lots,
        pattern=pattern,
        source=source,
        model=settings.default_model,
    )
    return [YieldTrendPoint(**r) for r in rows]


@router.get(
    "/stats/patterns", response_model=list[PatternCount], summary="Defect-pattern distribution"
)
def patterns(
    session: SessionDep,
    settings: SettingsDep,
    source: LabelSource = "predicted",
    lot: Annotated[str | None, Query(max_length=64)] = None,
) -> list[PatternCount]:
    rows = repo.pattern_distribution(session, source=source, model=settings.default_model, lot=lot)
    return [PatternCount(**r) for r in rows]


@router.get("/models", response_model=list[ModelInfo], tags=["models"], summary="Loaded models")
def models(registry: RegistryDep) -> list[ModelInfo]:
    out = []
    for name in registry.names():
        m = registry.get(name).meta
        out.append(
            ModelInfo(
                name=name,
                kind=m["kind"],
                version=m["version"],
                val_macro_f1=m.get("val_macro_f1"),
                test_macro_f1=m.get("test_macro_f1"),
            )
        )
    return out
