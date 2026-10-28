"""ORM models. Works on SQLite (local / tests) and PostgreSQL (docker-compose)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, deferred, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Lot(Base):
    __tablename__ = "lots"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    # Numeric part of the lot name. WM-811K has no timestamps, so lot order is the only
    # available proxy for production sequence (documented as a limitation).
    lot_number: Mapped[int] = mapped_column(Integer, index=True)
    source: Mapped[str] = mapped_column(String(16), default="wm811k")
    # Denormalised aggregates, refreshed on ingest / upload, so lot lists never scan wafers.
    wafer_count: Mapped[int] = mapped_column(Integer, default=0)
    mean_yield: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    wafers: Mapped[list[Wafer]] = relationship(back_populates="lot", order_by="Wafer.wafer_index")


class Wafer(Base):
    __tablename__ = "wafers"
    __table_args__ = (UniqueConstraint("lot_id", "wafer_index", name="uq_wafer_lot_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    lot_id: Mapped[int] = mapped_column(ForeignKey("lots.id", ondelete="CASCADE"), index=True)
    wafer_index: Mapped[int] = mapped_column(Integer)
    rows: Mapped[int] = mapped_column(Integer)
    cols: Mapped[int] = mapped_column(Integer)
    n_die: Mapped[int] = mapped_column(Integer)
    n_pass: Mapped[int] = mapped_column(Integer)
    n_fail: Mapped[int] = mapped_column(Integer)
    yield_pct: Mapped[float] = mapped_column(Float, index=True)
    # zlib-compressed uint8 grid; deferred so list endpoints never load it.
    map_zlib: Mapped[bytes] = deferred(mapped_column(LargeBinary, nullable=False))
    true_label: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    split: Mapped[str | None] = mapped_column(String(8), nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="wm811k")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    lot: Mapped[Lot] = relationship(back_populates="wafers")
    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="wafer", cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("wafer_id", "model_name", name="uq_prediction_wafer_model"),
        Index("ix_predictions_model_label", "model_name", "label"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    wafer_id: Mapped[int] = mapped_column(ForeignKey("wafers.id", ondelete="CASCADE"), index=True)
    model_name: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(32))
    label: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    wafer: Mapped[Wafer] = relationship(back_populates="predictions")
