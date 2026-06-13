"""Typed application settings loaded from environment variables and .env files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration.

    Values are read from environment variables, then `.env` if present.
    Every component (API, worker, scripts) imports the same `Settings`,
    so the system has one source of truth.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_log_level: Literal["debug", "info", "warning", "error"] = "info"
    api_cors_origins: str = "http://localhost:3000"

    # Database
    database_url: str = "sqlite:///./echomind.db"

    # Vector store
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_path: str = "./qdrant_storage"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # Cache
    cache_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"

    # Models
    whisper_model: str = "tiny"
    text_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    audio_embed_model: str = "laion/clap-htsat-unfused"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model_cache_dir: str = "./models/cache"

    # LLM / RAG
    llm_provider: Literal["openai", "anthropic", "ollama", "mock"] = "mock"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_host: str = "http://localhost:11434"

    # Storage
    storage_backend: Literal["local", "s3"] = "local"
    storage_path: str = "./data"
    s3_bucket: str | None = None
    s3_endpoint: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # MLflow
    mlflow_tracking_uri: str = "./mlruns"

    # Telemetry
    enable_prometheus: bool = True
    prometheus_port: int = 9090

    # Embedding dims (kept in sync with model choices)
    text_embed_dim: int = Field(default=384, description="MiniLM-L6 dim")
    audio_embed_dim: int = Field(default=512, description="CLAP dim")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]

    @property
    def model_cache_path(self) -> Path:
        path = Path(self.model_cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached for hot path)."""
    return Settings()
