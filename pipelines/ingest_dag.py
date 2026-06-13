"""Airflow DAG for batch ingestion.

Drop this file into your Airflow `dags/` directory. The DAG scans the audio
folder for new files daily, ingests each one, and updates the recommender
index.

Designed for an environment where Airflow is already installed — if you only
have it locally without Airflow, the worker `watch` command does the same job
in a tight loop.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

try:
    from airflow import DAG  # type: ignore[import-not-found]
    from airflow.operators.python import PythonOperator  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    DAG = None  # type: ignore[assignment]
    PythonOperator = None  # type: ignore[assignment]


AUDIO_DIR = Path("/data/audio")


def _scan_and_ingest(**_context) -> None:
    from echomind_worker.ingest import ingest_file

    for entry in AUDIO_DIR.iterdir():
        if not entry.is_file() or entry.suffix.lower() not in {".mp3", ".m4a", ".wav"}:
            continue
        slug = entry.stem.lower().replace(" ", "-")
        ingest_file(audio_path=entry, slug=slug, title=entry.stem)


def _refresh_recommender(**_context) -> None:
    from echomind_ml.recommender import main as recommender_main

    recommender_main()


default_args = {
    "owner": "echomind",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


if DAG is not None:
    with DAG(
        dag_id="echomind_ingest",
        description="Ingest new audiobooks and refresh the recommender index",
        start_date=datetime(2026, 1, 1),
        schedule="@daily",
        catchup=False,
        default_args=default_args,
        tags=["echomind", "ml"],
    ) as dag:
        ingest_task = PythonOperator(task_id="ingest", python_callable=_scan_and_ingest)
        recommender_task = PythonOperator(
            task_id="refresh_recommender", python_callable=_refresh_recommender
        )
        ingest_task >> recommender_task
