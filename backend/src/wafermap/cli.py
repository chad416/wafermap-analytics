"""Command-line entry point: ``wafermap <command>``.

Commands mirror the pipeline stages so each is reproducible on its own::

    wafermap prepare            # raw pickle -> data/processed/wafers.parquet + splits
    wafermap features           # engineered features for the labelled subset
    wafermap train-baseline     # LightGBM on engineered features
    wafermap train-cnn          # small CNN, exported to ONNX
    wafermap ingest             # processed parquet -> database (+ batch predictions)
"""

from __future__ import annotations

import argparse
import logging
import sys

from wafermap.config import get_settings


def _cmd_prepare(args: argparse.Namespace) -> None:
    from wafermap.data.prepare import prepare

    s = get_settings()
    summary = prepare(s.raw_dir, s.processed_dir, s.reports_dir, scope=args.scope, seed=s.seed)
    logging.info("labelled=%s rows_written=%s", summary["labelled_wafers"], summary["rows_written"])


def _cmd_features(args: argparse.Namespace) -> None:
    from wafermap.features.build import build_feature_table

    s = get_settings()
    build_feature_table(s.processed_dir, n_jobs=args.jobs)


def _cmd_train_baseline(args: argparse.Namespace) -> None:
    from wafermap.ml.train_baseline import train_baseline

    s = get_settings()
    train_baseline(s.processed_dir, s.model_dir, s.reports_dir, seed=s.seed)


def _cmd_train_cnn(args: argparse.Namespace) -> None:
    from wafermap.ml.train_cnn import train_cnn

    s = get_settings()
    train_cnn(
        s.processed_dir,
        s.model_dir,
        s.reports_dir,
        seed=s.seed,
        epochs=args.epochs,
        threads=args.threads,
    )


def _cmd_ingest(args: argparse.Namespace) -> None:
    from wafermap.db.ingest import ingest

    s = get_settings()
    ingest(s, limit_lots=args.limit_lots, predict=not args.no_predict, if_empty=args.if_empty)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wafermap", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prepare", help="clean + split the raw WM-811K pickle")
    sp.add_argument("--scope", choices=["labelled-lots", "labelled", "all"], default="labelled-lots")
    sp.set_defaults(func=_cmd_prepare)

    sp = sub.add_parser("features", help="compute engineered features")
    sp.add_argument("--jobs", type=int, default=-1)
    sp.set_defaults(func=_cmd_features)

    sp = sub.add_parser("train-baseline", help="train the LightGBM baseline")
    sp.set_defaults(func=_cmd_train_baseline)

    sp = sub.add_parser("train-cnn", help="train the CNN and export ONNX")
    sp.add_argument("--epochs", type=int, default=30)
    sp.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    sp.set_defaults(func=_cmd_train_cnn)

    sp = sub.add_parser("ingest", help="load processed data into the database")
    sp.add_argument("--limit-lots", type=int, default=None, help="only ingest the first N lots")
    sp.add_argument("--no-predict", action="store_true", help="skip batch predictions")
    sp.add_argument("--if-empty", action="store_true", help="no-op when the DB already has wafers")
    sp.set_defaults(func=_cmd_ingest)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
