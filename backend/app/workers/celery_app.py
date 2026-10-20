"""Celery application. Without Redis, jobs run in-process (see app.workers.tasks.dispatch)."""

from celery import Celery

from app.core.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "alignforge",
    broker=_settings.redis_url or "memory://",
    backend=_settings.redis_url or "cache+memory://",
    include=["app.workers.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # training jobs are long; don't hoard them
    task_routes={"alignforge.train_dpo": {"queue": "training"}, "alignforge.judge_samples": {"queue": "default"}},
    task_default_queue="default",
)
