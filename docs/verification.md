# Verification log

Evidence that the quality gates and the end-to-end stack work, captured on the development laptop
(Windows 11, Python 3.12.10, Node 24, Docker Desktop 29.6) on 2026-10-29. Re-run with
`make lint`, `make test`, `npm run build` and `docker compose up --build`.

## Quality gates (final commit)

```text
$ uv run ruff check . && uv run ruff check --config pyproject.toml ../scripts
All checks passed!
All checks passed!

$ uv run ruff format --check . && uv run ruff format --check --config pyproject.toml ../scripts
43 files already formatted
4 files already formatted

$ uv run mypy
Success: no issues found in 34 source files

$ uv run pytest
.......................................................                  [100%]
55 passed in 48.63s

$ npm run lint
(no problems)

$ npm run format:check
Checking formatting...
All matched files use Prettier code style!

$ npm run typecheck
(no errors)

$ npm test
 Test Files  4 passed (4)
      Tests  9 passed (9)

$ npm run build
✓ built in 296ms

$ git diff --check HEAD
(clean)
```

## End-to-end: `docker compose down -v && docker compose up --build -d`

```text
$ docker compose ps -a
api   running  Up 2 minutes (healthy)
db    running  Up 2 minutes (healthy)
seed  exited   Exited (0)
web   running  Up 2 minutes (healthy)

$ docker compose logs seed | tail -2
INFO wafermap.db.ingest:   244841 / 244841 wafers (163 s)
INFO wafermap.db.ingest: ingest finished in 163.6 s

$ curl -s http://127.0.0.1:8080/api/v1/health
{"status":"ok","database":true,"models":["cnn","baseline"],"version":"0.1.0"}

$ curl -s http://127.0.0.1:8080/api/v1/stats/summary
{"lots":10762,"wafers":244841,"labelled_wafers":172946,"predicted_wafers":244841,"uploaded_wafers":0,
 "mean_yield":87.711,"predicted_defect_share_pct":18.093,"default_model":"cnn"}

$ curl -s -X POST "http://127.0.0.1:8080/api/v1/predict/upload?model=baseline" -F file=@frontend/public/samples/edge-ring.csv
-> label Edge-Ring, confidence 0.998 (run on the empty-database stack before the CNN existed)
```

Response times through nginx with all wafers loaded: `/lots` 0.26 s, `/lots?pattern=Scratch` 0.06 s,
`/stats/summary` 0.30 s, `/stats/yield-trend` 0.13 s, `/stats/patterns` 0.07 s.

## Local fallback without Docker (SQLite)

`wafermap migrate && wafermap ingest` loaded the same 244,841 wafers into `data/wafermap.db`
(160 MB, 1,395 s on the host). The API on SQLite returned identical KPIs to the PostgreSQL stack, and
the Vite dev server proxied `/api` correctly.

## Not verified here

- The GitHub Actions workflow has not been run (the repository has not been pushed). Its steps
  mirror the commands above, which pass locally, but the Postgres CI job and the compose smoke test
  have only been exercised locally in their Docker equivalents.
- The Makefile was not executed with GNU make (Windows host); each target's commands were run directly.
