FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install workspace packages in editable mode
COPY packages/core /app/packages/core
COPY packages/ml /app/packages/ml
COPY services/api /app/services/api

RUN pip install --upgrade pip && \
    pip install -e packages/core && \
    pip install -e packages/ml && \
    pip install -e services/api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "echomind_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
