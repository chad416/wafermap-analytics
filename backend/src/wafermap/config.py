"""Runtime configuration (12-factor: everything overridable via environment / ``.env``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/src/wafermap/config.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WAFERMAP_", env_file=".env", extra="ignore")

    database_url: str = Field(default=f"sqlite:///{(REPO_ROOT / 'data' / 'wafermap.db').as_posix()}")
    data_dir: Path = REPO_ROOT / "data"
    model_dir: Path = REPO_ROOT / "models"
    reports_dir: Path = REPO_ROOT / "reports"
    # Model used for stored predictions and as the default for /predict.
    default_model: str = "cnn"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    max_upload_bytes: int = 2 * 1024 * 1024
    seed: int = 42

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


@lru_cache
def get_settings() -> Settings:
    return Settings()
