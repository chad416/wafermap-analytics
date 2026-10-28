from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from synthetic import center, edge_ring, random_fail, scratch
from wafermap import wafer
from wafermap.api.deps import settings_dep
from wafermap.api.main import create_app
from wafermap.config import REPO_ROOT, Settings
from wafermap.db.migrate import upgrade
from wafermap.db.models import Lot, Prediction, Wafer
from wafermap.db.repository import refresh_lot_stats
from wafermap.db.session import get_sessionmaker

MODEL_DIR = REPO_ROOT / "models"
HAVE_MODELS = (MODEL_DIR / "cnn" / "model.onnx").exists() and (MODEL_DIR / "baseline" / "model.txt").exists()


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    db = tmp_path_factory.mktemp("db") / "test.db"
    s = Settings(database_url=f"sqlite:///{db.as_posix()}", model_dir=MODEL_DIR, data_dir=Path(db.parent))
    upgrade(s.database_url)  # the test DB is built by the real migrations
    _seed(s)
    return s


def _seed(s: Settings) -> None:
    """3 lots x 4 wafers with known labels / predictions."""
    makers = {"Center": center, "Edge-Ring": edge_ring, "Scratch": scratch, "Random": random_fail}
    with get_sessionmaker(s.database_url)() as session:
        for lot_no in (1, 2, 3):
            lot = Lot(name=f"lot{lot_no}", lot_number=lot_no)
            session.add(lot)
            session.flush()
            for idx, (label, make) in enumerate(makers.items(), start=1):
                m = make()
                st = wafer.stats(m)
                w = Wafer(lot_id=lot.id, wafer_index=idx, rows=m.shape[0], cols=m.shape[1], n_die=st.n_die,
                          n_pass=st.n_pass, n_fail=st.n_fail, yield_pct=round(st.yield_pct, 3),
                          map_zlib=wafer.encode(m), true_label=label if lot_no != 3 else None,
                          split="test" if lot_no != 3 else None)
                pred = "none" if label == "Random" else label
                w.predictions = [Prediction(model_name="cnn", model_version="test", label=pred, confidence=0.9,
                                            probabilities={pred: 0.9})]
                session.add(w)
        session.flush()
        refresh_lot_stats(session)
        session.commit()


@pytest.fixture(scope="session")
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    app.dependency_overrides[settings_dep] = lambda: settings
    with TestClient(app) as c:
        yield c


@pytest.fixture
def session(settings: Settings) -> Iterator[Session]:
    with get_sessionmaker(settings.database_url)() as s:
        yield s
