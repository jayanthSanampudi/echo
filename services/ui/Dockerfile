FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY packages/core /app/packages/core
COPY services/ui /app/services/ui

RUN pip install --upgrade pip && \
    pip install -e packages/core && \
    pip install -e services/ui

EXPOSE 8501

CMD ["streamlit", "run", "/app/services/ui/echomind_ui/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]
