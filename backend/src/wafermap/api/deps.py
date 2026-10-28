"""FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Query, Request
from sqlalchemy.orm import Session

from wafermap.config import Settings, get_settings
from wafermap.db.session import session_scope
from wafermap.ml.predict import ModelRegistry


def settings_dep() -> Settings:
    return get_settings()


def db_session(settings: Annotated[Settings, Depends(settings_dep)]) -> Iterator[Session]:
    yield from session_scope(settings.database_url)


def registry_dep(request: Request) -> ModelRegistry:
    registry: ModelRegistry = request.app.state.registry
    return registry


class Pagination:
    def __init__(
        self,
        page: Annotated[int, Query(ge=1, description="1-based page number")] = 1,
        page_size: Annotated[int, Query(ge=1, le=200)] = 25,
    ) -> None:
        self.page = page
        self.page_size = page_size

    def pages(self, total: int) -> int:
        return max(1, -(-total // self.page_size))


SessionDep = Annotated[Session, Depends(db_session)]
SettingsDep = Annotated[Settings, Depends(settings_dep)]
RegistryDep = Annotated[ModelRegistry, Depends(registry_dep)]
PaginationDep = Annotated[Pagination, Depends()]
