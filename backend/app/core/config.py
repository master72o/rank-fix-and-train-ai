"""Runtime configuration loaded from environment variables / .env files."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    app_name: str = "AlignForge"
    api_prefix: str = "/api/v1"

    # SQLite works out of the box; docker-compose switches this to PostgreSQL 16 + pgvector.
    database_url: str = "sqlite+aiosqlite:///./alignforge.db"
    auto_create_tables: bool = True

    # When REDIS_URL is set and USE_CELERY=true, jobs go to Celery workers.
    # Otherwise they run in-process as FastAPI background tasks.
    redis_url: str | None = None
    use_celery: bool = False

    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Models
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    embedding_backend: Literal["hashing", "sentence-transformers"] = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    generator_backend: Literal["none", "local", "litellm"] = "none"
    generator_litellm_model: str | None = None
    generator_max_new_tokens: int = 384
    # "heuristic" = offline rubric baseline; anything else is passed to LiteLLM (e.g. "gpt-4o-mini").
    judge_model: str = "heuristic"
    judge_temperature: float = 0.0

    # Active learning
    dedupe_threshold: float = 0.95
    priority_entropy_weight: float = 0.5
    priority_divergence_weight: float = 0.5
    human_review_threshold: float = 0.35

    training_output_dir: str = "./artifacts/training"

    @property
    def celery_enabled(self) -> bool:
        return bool(self.use_celery and self.redis_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
