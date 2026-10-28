"""Programmatic Alembic upgrade (used by the CLI, the container entrypoint and the tests)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

ALEMBIC_INI = Path(__file__).resolve().parents[3] / "alembic.ini"


def upgrade(database_url: str, revision: str = "head") -> None:
    cfg = Config(str(ALEMBIC_INI))
    cfg.attributes["database_url"] = database_url
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, revision)
