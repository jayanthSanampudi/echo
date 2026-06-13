.PHONY: help install dev test lint typecheck fmt seed api worker ui up down clean

help:
	@echo "EchoMind — common commands"
	@echo ""
	@echo "  make install     Install all packages (uv sync)"
	@echo "  make dev         Install with dev extras"
	@echo "  make seed        Download + index a tiny LibriVox sample"
	@echo "  make api         Run the FastAPI service on :8000"
	@echo "  make worker      Run the ingestion worker (CLI)"
	@echo "  make ui          Run the Streamlit UI on :8501"
	@echo "  make test        Run the pytest suite"
	@echo "  make lint        Run ruff"
	@echo "  make typecheck   Run mypy"
	@echo "  make fmt         Auto-format with ruff"
	@echo "  make up          docker compose up --build"
	@echo "  make down        docker compose down -v"
	@echo "  make clean       Remove caches and build artifacts"

install:
	uv sync

dev:
	uv sync --all-extras

seed:
	uv run python scripts/seed_data.py

api:
	uv run uvicorn echomind_api.main:app --reload --host 0.0.0.0 --port 8000

worker:
	uv run python -m echomind_worker.cli

ui:
	uv run streamlit run services/ui/echomind_ui/app.py

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy packages services

fmt:
	uv run ruff format .
	uv run ruff check --fix .

up:
	docker compose up --build

down:
	docker compose down -v

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
