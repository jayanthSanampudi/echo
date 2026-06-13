FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY packages/core /app/packages/core
COPY packages/ml /app/packages/ml
COPY services/worker /app/services/worker

RUN pip install --upgrade pip && \
    pip install -e packages/core && \
    pip install -e packages/ml && \
    pip install -e services/worker

ENTRYPOINT ["echomind-worker"]
CMD ["--help"]
