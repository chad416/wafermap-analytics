# syntax=docker/dockerfile:1
# API / seed image. Build context = repository root (needs models/).
#   docker build -f backend/Dockerfile -t wafermap-api .
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# libgomp1: OpenMP runtime required by LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.11.24

WORKDIR /app/backend

# Dependency layer first (cached unless the lock file changes). Runtime deps only:
# no dev tools and no "ml" extra, so no PyTorch - the CNN is served through ONNX Runtime.
COPY backend/pyproject.toml backend/uv.lock ./
# BuildKit cache mount keeps uv's download cache out of the image layers.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY backend/ ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

COPY models/ /app/models/

ENV WAFERMAP_MODEL_DIR=/app/models \
    WAFERMAP_DATA_DIR=/app/data \
    WAFERMAP_REPORTS_DIR=/app/reports

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown app /app/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=2)"

# Apply migrations, then serve. Two workers: inference is CPU-bound and the box is small.
CMD ["sh", "-c", "wafermap migrate && exec uvicorn wafermap.api.main:app --host 0.0.0.0 --port 8000 --workers 2 --proxy-headers"]
