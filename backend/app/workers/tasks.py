"""
Background jobs: LLM-judge passes and DPO training + evaluation.

The job bodies are plain async functions so they can run either
* inside a Celery worker (`USE_CELERY=true` + `REDIS_URL`), with a fresh NullPool engine per task, or
* in-process as FastAPI background tasks (default local mode).
"""

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from fastapi import BackgroundTasks
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.database import SessionLocal, make_engine, make_session_factory
from app.models.domain import (
    Annotation,
    AnnotatorType,
    Dataset,
    PreferenceChoice,
    PromptSample,
    RunStatus,
    SampleStatus,
    TrainingRun,
)
from app.services.dpo_trainer import (
    DPOTrainingConfig,
    TrainingDependencyError,
    build_preference_pairs,
    ml_available,
    run_dpo_training,
)
from app.services.eval_harness import build_hf_generators, evaluate_win_rate
from app.services.ingestion import refresh_sample_status
from app.services.llm_judge import DEFAULT_RUBRIC, evaluate_with_position_swap, get_judge
from app.workers.celery_app import celery_app

SessionFactory = async_sessionmaker[AsyncSession]


# ----------------------------------------------------------------------------- judge job

async def judge_samples_job(sample_ids: list[str], session_factory: SessionFactory | None = None) -> int:
    factory = session_factory or SessionLocal
    judge = get_judge()
    done = 0
    async with factory() as session:
        for sid in sample_ids:
            sample = await session.get(PromptSample, uuid.UUID(str(sid)))
            if sample is None:
                continue
            dataset = await session.get(Dataset, sample.dataset_id)
            rubric = (dataset.rubric_definition if dataset else None) or DEFAULT_RUBRIC
            t0 = time.perf_counter()
            verdict = await asyncio.to_thread(
                evaluate_with_position_swap, judge, sample.user_prompt, sample.candidate_a, sample.candidate_b, rubric
            )
            # Re-running the same judge replaces its previous verdict (idempotent).
            await session.execute(delete(Annotation).where(
                Annotation.sample_id == sample.id,
                Annotation.annotator_type == AnnotatorType.LLM_JUDGE,
                Annotation.annotator_id == judge.name,
            ))
            session.add(Annotation(
                sample_id=sample.id,
                annotator_id=judge.name,
                annotator_type=AnnotatorType.LLM_JUDGE,
                preference_choice=PreferenceChoice(verdict.preference),
                rubric_scores=verdict.rubric_scores,
                critique_rationale=verdict.rationale,
                position_consistent=verdict.position_consistent,
                time_spent_ms=int((time.perf_counter() - t0) * 1000),
                error_flags=[],
            ))
            await session.flush()
            await refresh_sample_status(session, sample)
            await session.commit()
            done += 1
    return done


# ----------------------------------------------------------------------------- training job

async def train_dpo_job(run_id: str, session_factory: SessionFactory | None = None) -> None:
    factory = session_factory or SessionLocal
    settings = get_settings()
    async with factory() as session:
        run = await session.get(TrainingRun, uuid.UUID(str(run_id)))
        if run is None:
            return
        run.status, run.started_at = RunStatus.RUNNING, datetime.now(UTC)
        await session.commit()
        try:
            cfg_in = dict(run.config)
            samples = (await session.execute(
                select(PromptSample).where(PromptSample.dataset_id == run.dataset_id)
                .options(selectinload(PromptSample.annotations))
                .order_by(PromptSample.active_learning_priority.desc())
            )).scalars().all()
            pairs = build_preference_pairs(
                ((s, s.annotations) for s in samples), include_judge_labels=cfg_in.get("include_judge_labels", False)
            )
            paired_ids = {p.sample_id for p in pairs}
            holdout_n = int(cfg_in.get("eval_holdout", 8))
            eval_prompts = [(s.system_prompt, s.user_prompt) for s in samples if str(s.id) not in paired_ids][:holdout_n]
            run.num_pairs = len(pairs)
            await session.commit()

            if len(pairs) < 2:
                raise ValueError(f"Need at least 2 preference pairs to train; found {len(pairs)}. Annotate more samples.")
            if not ml_available():
                raise TrainingDependencyError(
                    "ML extra not installed in this worker. Run: pip install -e 'backend[ml]' (or build the "
                    "backend image with INSTALL_ML=true)."
                )

            cfg = DPOTrainingConfig(
                base_model=run.base_model,
                output_dir=str(Path(settings.training_output_dir) / str(run.id)),
                **{k: cfg_in[k] for k in (
                    "beta", "learning_rate", "num_train_epochs", "per_device_batch_size",
                    "gradient_accumulation_steps", "max_length", "lora_r", "lora_alpha", "lora_dropout", "use_qlora",
                ) if k in cfg_in},
            )
            logs: list[dict] = []
            train_task = asyncio.create_task(asyncio.to_thread(run_dpo_training, pairs, cfg, logs.append))
            while not train_task.done():
                await asyncio.sleep(2)
                run.metrics = {"log_history": list(logs), "phase": "training"}
                await session.commit()
            result = await train_task
            run.adapter_path = result["adapter_path"]
            run.metrics = {
                "log_history": list(logs), "phase": "evaluating",
                "train_loss": result["train_loss"], "train_runtime_s": result["train_runtime_s"],
            }
            await session.commit()

            if eval_prompts:
                baseline_fn, tuned_fn = await asyncio.to_thread(build_hf_generators, run.base_model, run.adapter_path)
                report = await asyncio.to_thread(evaluate_win_rate, eval_prompts, baseline_fn, tuned_fn, get_judge())
                run.eval_report = report.to_dict()
            else:
                run.eval_report = {"message": "No held-out prompts available (every sample was used for training)."}

            for s in samples:
                if str(s.id) in paired_ids:
                    s.status = SampleStatus.EXPORTED
            run.metrics = {**run.metrics, "phase": "done"}
            run.status = RunStatus.SUCCEEDED
        except Exception as exc:  # surface every failure to the UI
            run.status = RunStatus.FAILED
            run.error = f"{type(exc).__name__}: {exc}"
            run.metrics = {**(run.metrics or {}), "phase": "failed", "traceback": traceback.format_exc()[-4000:]}
        finally:
            run.finished_at = datetime.now(UTC)
            await session.commit()


# ----------------------------------------------------------------------------- celery wrappers

def _run_in_fresh_loop(job: Callable[..., Awaitable], *args) -> object:
    async def _main():
        engine = make_engine(null_pool=True)
        try:
            return await job(*args, session_factory=make_session_factory(engine))
        finally:
            await engine.dispose()

    return asyncio.run(_main())


@celery_app.task(name="alignforge.judge_samples")
def judge_samples_task(sample_ids: list[str]) -> int:
    return _run_in_fresh_loop(judge_samples_job, sample_ids)  # type: ignore[return-value]


@celery_app.task(name="alignforge.train_dpo")
def train_dpo_task(run_id: str) -> None:
    _run_in_fresh_loop(train_dpo_job, run_id)


def dispatch_judge(background: BackgroundTasks, sample_ids: list[str]) -> str:
    if get_settings().celery_enabled:
        judge_samples_task.delay(sample_ids)
        return "celery"
    background.add_task(judge_samples_job, sample_ids)
    return "background"


def dispatch_training(background: BackgroundTasks, run_id: str) -> str:
    if get_settings().celery_enabled:
        train_dpo_task.delay(run_id)
        return "celery"
    background.add_task(train_dpo_job, run_id)
    return "background"
