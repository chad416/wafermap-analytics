"""Pydantic request / response models (these define the OpenAPI schema)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PatternName = Literal["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
ModelName = Literal["cnn", "baseline"]
LabelSource = Literal["true", "predicted"]


class Page[T](BaseModel):
    items: list[T]
    total: int
    page: int
    page_size: int
    pages: int


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    model_name: str
    model_version: str
    label: str
    confidence: float
    probabilities: dict[str, float]


class WaferSummary(BaseModel):
    id: int
    lot_name: str
    wafer_index: int
    rows: int
    cols: int
    n_die: int
    n_fail: int
    yield_pct: float
    true_label: str | None
    split: str | None
    source: str
    prediction: PredictionOut | None = Field(None, description="Stored prediction of the default model")


class WaferDetail(WaferSummary):
    predictions: list[PredictionOut]


class WaferMapOut(BaseModel):
    id: int
    rows: int
    cols: int
    grid: list[list[int]] = Field(description="Row-major die bins: 0 = no die, 1 = pass, 2 = fail")


class LotOut(BaseModel):
    name: str
    lot_number: int
    source: str
    wafer_count: int
    mean_yield: float | None
    pattern_counts: dict[str, int] = Field(description="Wafers per pattern (source/model as requested)")
    top_defect: str | None = Field(description="Most frequent non-'none' pattern in the lot")


class LotDetail(LotOut):
    wafers: list[WaferSummary]


class YieldTrendPoint(BaseModel):
    bucket: int
    lot_number_from: int
    lot_number_to: int
    lots: int
    wafers: int
    mean_yield: float
    min_yield: float


class PatternCount(BaseModel):
    pattern: str
    count: int
    share_pct: float
    mean_yield: float | None


class Summary(BaseModel):
    lots: int
    wafers: int
    labelled_wafers: int
    predicted_wafers: int
    uploaded_wafers: int
    mean_yield: float | None
    predicted_defect_share_pct: float | None
    default_model: str


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    kind: str
    version: str
    val_macro_f1: float | None
    test_macro_f1: float | None


class PredictRequest(BaseModel):
    grid: list[list[int]] = Field(
        min_length=3, max_length=512, description="2-D die-bin grid: 0 = no die, 1 = pass, 2 = fail"
    )


class WaferStatsOut(BaseModel):
    rows: int
    cols: int
    n_die: int
    n_fail: int
    yield_pct: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    stats: WaferStatsOut
    latency_ms: float
    wafer_id: int | None = Field(None, description="Set when the map was saved (save=true)")
    grid: list[list[int]] | None = Field(None, description="Parsed map, echoed for file uploads")


class Health(BaseModel):
    status: Literal["ok"]
    database: bool
    models: list[str]
    version: str
