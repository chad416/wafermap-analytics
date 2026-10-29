"""Load ``data/processed/wafers.parquet`` into the database and store batch predictions."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sqlalchemy import func, insert, select, text

from wafermap import wafer
from wafermap.config import Settings
from wafermap.db.models import Lot, Prediction, Wafer
from wafermap.db.repository import refresh_lot_stats
from wafermap.db.session import get_sessionmaker
from wafermap.ml.predict import ModelRegistry, top

log = logging.getLogger(__name__)
CHUNK = 5000


def ingest(
    settings: Settings,
    *,
    limit_lots: int | None = None,
    predict: bool = True,
    if_empty: bool = False,
) -> int:
    path = settings.processed_dir / "wafers.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `wafermap prepare` first")
    sm = get_sessionmaker(settings.database_url)
    with sm() as s:
        existing = s.scalar(select(func.count(Wafer.id))) or 0
        if existing:
            if if_empty:
                log.info("database already holds %d wafers - skipping ingest", existing)
                return 0
            raise RuntimeError(
                f"database already holds {existing} wafers; ingest expects an empty "
                "database (use --if-empty to make it a no-op)"
            )

    table = pq.read_table(path).sort_by([("lot_number", "ascending"), ("wafer_index", "ascending")])
    if limit_lots is not None:
        keep = pc.unique(table["lot_name"])[:limit_lots]
        table = table.filter(pc.is_in(table["lot_name"], value_set=keep))
    n = int(table.num_rows)
    log.info("ingesting %d wafers from %s", n, path)

    classifier = None
    if predict:
        registry = ModelRegistry(settings.model_dir)
        if settings.default_model in registry.names():
            classifier = registry.get(settings.default_model)
        else:
            log.warning(
                "default model %s unavailable - wafers ingested without predictions",
                settings.default_model,
            )

    t0 = time.perf_counter()
    with sm() as s:
        lot_ids = _upsert_lots(
            s, table.column("lot_name").to_pylist(), table.column("lot_number").to_pylist()
        )
        for start in range(0, n, CHUNK):
            batch = table.slice(start, CHUNK).to_pydict()
            rows = _wafer_rows(batch, lot_ids)
            ids = list(
                s.scalars(insert(Wafer).returning(Wafer.id, sort_by_parameter_order=True), rows)
            )
            if classifier is not None:
                maps = [
                    wafer.decode(b, r, c)
                    for b, r, c in zip(batch["map_zlib"], batch["rows"], batch["cols"], strict=True)
                ]
                preds = top(classifier.predict_proba(maps))
                s.execute(
                    insert(Prediction),
                    [
                        {
                            "wafer_id": wid,
                            "model_name": classifier.name,
                            "model_version": classifier.version,
                            "label": label,
                            "confidence": conf,
                            "probabilities": probs,
                        }
                        for wid, (label, conf, probs) in zip(ids, preds, strict=True)
                    ],
                )
            s.commit()
            log.info(
                "  %d / %d wafers (%.0f s)", min(start + CHUNK, n), n, time.perf_counter() - t0
            )
        refresh_lot_stats(s)
        s.commit()
        # Refresh planner statistics after the bulk load. Without them SQLite answers the
        # per-lot pattern counts by scanning every prediction (0.26 s) instead of starting
        # from the lot index (2 ms); PostgreSQL would otherwise wait for autovacuum.
        s.execute(text("ANALYZE"))
        s.commit()
    log.info("ingest finished in %.1f s", time.perf_counter() - t0)
    return n


def _upsert_lots(s: Any, names: list[str], numbers: list[int]) -> dict[str, int]:
    wanted = dict(zip(names, numbers, strict=True))
    have = set(s.scalars(select(Lot.name)))
    new = [
        {"name": k, "lot_number": v, "source": "wm811k"} for k, v in wanted.items() if k not in have
    ]
    for i in range(0, len(new), CHUNK):
        s.execute(insert(Lot), new[i : i + CHUNK])
    s.flush()
    return dict(s.execute(select(Lot.name, Lot.id)).all())


def _wafer_rows(batch: dict[str, list[Any]], lot_ids: dict[str, int]) -> list[dict[str, Any]]:
    n_die = np.asarray(batch["n_die"])
    n_fail = np.asarray(batch["n_fail"])
    yld = 100.0 * (n_die - n_fail) / n_die
    return [
        {
            "lot_id": lot_ids[batch["lot_name"][i]],
            "wafer_index": batch["wafer_index"][i],
            "rows": batch["rows"][i],
            "cols": batch["cols"][i],
            "n_die": int(n_die[i]),
            "n_pass": int(n_die[i] - n_fail[i]),
            "n_fail": int(n_fail[i]),
            "yield_pct": round(float(yld[i]), 3),
            "map_zlib": batch["map_zlib"][i],
            "true_label": batch["label"][i],
            "split": batch["split"][i],
            "source": "wm811k",
        }
        for i in range(len(n_die))
    ]
