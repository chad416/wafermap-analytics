# Verification log

What was run on the final commit, from a fresh `git clone` of the repository, on the development
laptop (Windows 11, Intel Core i5-13450HX, 16 logical CPUs, Python 3.12.10 via uv 0.11.24,
Node 24.15, Docker Desktop 29.6) on 2026-10-29. Re-run with `make lint`, `make test`,
`npm run build` and `docker compose up --build`.

## Quality gates

```text
$ uv run ruff check . && uv run ruff check --config pyproject.toml ../scripts
All checks passed!
All checks passed!

$ uv run ruff format --check . && uv run ruff format --check --config pyproject.toml ../scripts
43 files already formatted
5 files already formatted

$ uv run mypy
Success: no issues found in 34 source files

$ uv run pytest
59 passed, 3 warnings
(the warnings are third-party deprecations: Starlette's TestClient on httpx, and the legacy
TorchScript ONNX exporter used in the export test)

$ uv run pre-commit run --all-files --config ../.pre-commit-config.yaml
all 11 hooks passed

$ npm ci && npm run lint && npm run format:check && npm run typecheck
(no problems)

$ npm test
 Test Files  4 passed (4)
      Tests  10 passed (10)

$ npm run build
built; route chunks 0.4 to 3.3 kB gzip, overview chunk 108 kB gzip, core 84 kB gzip

$ git diff --check HEAD
(clean)
```

## Data pipeline from a clean checkout

```text
$ python scripts/fetch_dataset.py        # file staged in data/raw/, hash verified
data/raw/LSWMD.pkl already present and verified

$ uv sync --extra ml && uv run wafermap prepare
labelled=172946 rows_written=244841       (about 4 min, peak working set 2.72 GiB)
```

`data/processed/wafers.parquet` came out byte-identical to the one the models were trained on,
and `reports/` was unchanged, so the split is reproducible. The mirror URL in the fetch script
was checked separately: it serves 2,095,505,977 bytes with the pinned SHA-256 as its ETag.

## End-to-end: `docker compose up --build -d --wait`

```text
$ docker compose ps -a
api   Up (healthy)    db   Up (healthy)    web   Up (healthy)    seed  Exited (0)

$ docker compose logs seed | tail -1
INFO wafermap.db.ingest: ingest finished in 274.9 s      (another job was using the CPU)

$ curl -s http://127.0.0.1:8080/api/v1/stats/summary
{"lots":10762,"wafers":244841,"labelled_wafers":172946,"predicted_wafers":244841,
 "uploaded_wafers":0,"mean_yield":87.711,"predicted_defect_share_pct":18.093,"default_model":"cnn"}

$ curl -s http://127.0.0.1:8080/api/v1/models
cnn f303ccc7a3f7 (val 0.8928, test 0.8988), baseline e63edfe3cdb3 (val 0.8889, test 0.8869)

$ curl -s -X POST "http://127.0.0.1:8000/api/v1/predict/upload?model=baseline" -F file=@frontend/public/samples/scratch.csv
-> Scratch, 15 to 20 ms server-side, already on the first request after startup (models are warmed up)
```

An earlier idle run of the same stack ingested the 244,841 wafers in 164 s.

Screenshots were retaken from this stack with `scripts/screenshots.py`. Four came out
byte-identical; the two that display a server-side latency (wafer detail, upload) changed.

## Local fallback without Docker or data (SQLite)

On a clean checkout with no `data/` directory, `wafermap migrate` creates `data/wafermap.db`, and
the API started on it answers `/health`, `/stats/summary` (all zeros), `/lots` (empty page) and
`/predict` (baseline, first request 12 ms). Invalid input returns 422 with a message (unknown
model name, bin code 3).

## Results reproduced from the committed models

`scripts/compare_models.py` re-ran both committed models on the 25,936 test wafers: both
reproduce the confusion matrices stored in `reports/metrics/` exactly. `wafermap report` was run
twice for latency; the two runs agree within 3 %.

## Not verified here

- The GitHub Actions workflow has not run (the repository has not been pushed). Its steps are
  the commands above; the PostgreSQL test job and the compose smoke test were only exercised
  through the local Docker stack.
- GNU make is not installed on the Windows host, so each Makefile target's commands were run
  directly.
- The dataset download itself was not repeated (the file was copied in and verified by hash).
- The CNN was not retrained (about 2 h on CPU); its metrics come from the committed run and were
  re-checked through the committed ONNX model as described above.
