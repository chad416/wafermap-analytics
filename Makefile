# Developer entry points. Works with GNU make (Linux/macOS, or Git Bash / WSL on Windows).
# Every target is a thin wrapper around a command documented in README.md.

UV      ?= uv
NPM     ?= npm
BACKEND := backend
FRONTEND:= frontend
RUN     := cd $(BACKEND) && $(UV) run

.PHONY: help install fetch-data data features train train-baseline train-cnn report migrate \
        ingest api web test test-backend test-frontend lint format typecheck build up down clean

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n",$$1,$$2}'

install:         ## install backend (incl. training extras) and frontend dependencies
	cd $(BACKEND) && $(UV) sync --extra ml
	cd $(FRONTEND) && $(NPM) ci

fetch-data:      ## download WM-811K (2.1 GB) and verify its SHA-256
	python scripts/fetch_dataset.py

data: fetch-data ## fetch + clean + split -> data/processed/wafers.parquet
	$(RUN) wafermap prepare

features:        ## engineered features for the baseline
	$(RUN) wafermap features

train-baseline: features ## train + evaluate the LightGBM baseline
	$(RUN) wafermap train-baseline

train-cnn:       ## train + evaluate the CNN, export ONNX (CPU, ~1-2 h)
	$(RUN) wafermap train-cnn

train: train-baseline train-cnn ## reproduce both models, their metrics and confusion matrices

report:          ## summary table, latency, figures, split-leakage check, paired comparison
	$(RUN) wafermap report
	$(RUN) python ../scripts/split_leakage_experiment.py
	$(RUN) python ../scripts/compare_models.py

migrate:         ## apply Alembic migrations (SQLite by default)
	$(RUN) wafermap migrate

ingest: migrate  ## load processed data into the DB with batch CNN predictions
	$(RUN) wafermap ingest

api: migrate     ## run the API with auto-reload on :8000
	$(RUN) uvicorn wafermap.api.main:app --reload --port 8000

web:             ## run the Vite dev server on :5173 (proxies /api to :8000)
	cd $(FRONTEND) && $(NPM) run dev

test: test-backend test-frontend ## all tests

test-backend:    ## pytest (SQLite)
	$(RUN) pytest

test-frontend:   ## vitest
	cd $(FRONTEND) && $(NPM) test

lint:            ## ruff + mypy + eslint + prettier check
	$(RUN) ruff check . && $(UV) run ruff check --config pyproject.toml ../scripts
	$(RUN) ruff format --check . && $(UV) run ruff format --check --config pyproject.toml ../scripts
	$(RUN) mypy
	cd $(FRONTEND) && $(NPM) run lint && $(NPM) run format:check

format:          ## auto-format backend and frontend
	$(RUN) ruff format .
	$(RUN) ruff check --fix .
	cd $(FRONTEND) && $(NPM) run format

typecheck:       ## mypy + tsc
	$(RUN) mypy
	cd $(FRONTEND) && $(NPM) run typecheck

build:           ## production frontend build + docker images
	cd $(FRONTEND) && $(NPM) run build
	docker compose build

up:              ## start the full stack (db, api, seed, web)
	docker compose up --build -d
	@echo "UI  http://localhost:8080   API http://localhost:8000/docs"

down:            ## stop the stack (keeps the database volume)
	docker compose down

clean:           ## remove build outputs and caches (not data/ or models/)
	rm -rf $(FRONTEND)/dist $(BACKEND)/.pytest_cache $(BACKEND)/.ruff_cache $(BACKEND)/.mypy_cache
