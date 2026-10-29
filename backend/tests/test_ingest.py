from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import func, select

from conftest import HAVE_MODELS, MODEL_DIR
from synthetic import center, edge_ring, scratch
from wafermap import wafer
from wafermap.config import Settings
from wafermap.db.ingest import ingest
from wafermap.db.migrate import upgrade
from wafermap.db.models import Lot, Prediction, Wafer
from wafermap.db.session import get_sessionmaker


def _write_parquet(processed: Path) -> None:
    rows = []
    for lot_no in (7, 3):  # deliberately out of order
        for idx, (make, label) in enumerate(
            [(center, "Center"), (edge_ring, None), (scratch, "Scratch")], start=1
        ):
            m = make()
            st = wafer.stats(m)
            rows.append(
                {
                    "lot_name": f"lot{lot_no}",
                    "lot_number": lot_no,
                    "wafer_index": idx,
                    "die_size": st.n_die,
                    "rows": m.shape[0],
                    "cols": m.shape[1],
                    "map_zlib": wafer.encode(m),
                    "n_die": st.n_die,
                    "n_fail": st.n_fail,
                    "label": label,
                    "original_split": None,
                    "split": "train" if label else None,
                }
            )
    processed.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), processed / "wafers.parquet")


@pytest.mark.parametrize(
    "predict",
    [
        False,
        pytest.param(True, marks=pytest.mark.skipif(not HAVE_MODELS, reason="models not present")),
    ],
)
def test_ingest_loads_wafers_and_lot_stats(tmp_path: Path, predict: bool) -> None:
    s = Settings(
        database_url=f"sqlite:///{(tmp_path / 'i.db').as_posix()}",
        data_dir=tmp_path,
        model_dir=MODEL_DIR,
    )
    _write_parquet(s.processed_dir)
    upgrade(s.database_url)
    assert ingest(s, predict=predict) == 6

    with get_sessionmaker(s.database_url)() as session:
        lots = session.scalars(select(Lot).order_by(Lot.lot_number)).all()
        assert [lot.name for lot in lots] == ["lot3", "lot7"]
        assert all(lot.wafer_count == 3 and lot.mean_yield is not None for lot in lots)
        w = session.scalar(select(Wafer).where(Wafer.true_label == "Center"))
        assert w is not None and w.yield_pct == pytest.approx(100 * w.n_pass / w.n_die, abs=1e-3)
        n_pred = session.scalar(select(func.count(Prediction.id)))
        assert n_pred == (6 if predict else 0)

    # a second ingest must not duplicate data
    assert ingest(s, if_empty=True) == 0
    with pytest.raises(RuntimeError, match="already holds"):
        ingest(s)


def test_ingest_missing_file(tmp_path: Path) -> None:
    s = Settings(database_url=f"sqlite:///{(tmp_path / 'm.db').as_posix()}", data_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        ingest(s)


def test_migrate_creates_missing_sqlite_directory(tmp_path: Path) -> None:
    # A fresh checkout has no data/ directory; migrating the default SQLite URL must still work.
    db = tmp_path / "not-yet-created" / "wafermap.db"
    upgrade(f"sqlite:///{db.as_posix()}")
    assert db.exists()
