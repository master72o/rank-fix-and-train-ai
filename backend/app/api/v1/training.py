"""API routes for launching and monitoring DPO training runs and preference-pair previews."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.database import get_session
from app.models.domain import Dataset, PromptSample, RunStatus, TrainingRun
from app.schemas.payloads import (
    Capabilities,
    PairPreview,
    TrainingRunCreate,
    TrainingRunOut,
)
from app.services.dpo_trainer import (
    build_preference_pairs,
    cuda_available,
    ml_available,
)
from app.workers.tasks import dispatch_training

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/capabilities", response_model=Capabilities)
async def get_system_capabilities():
    s = get_settings()
    return Capabilities(
        ml_available=ml_available(),
        cuda_available=cuda_available(),
        base_model=s.base_model,
        judge_model=s.judge_model,
        generator_backend=s.generator_backend,
        embedding_backend=s.embedding_backend,
        celery_enabled=s.celery_enabled,
    )


@router.get("/preview-pairs", response_model=PairPreview)
async def preview_pairs(
    dataset_id: uuid.UUID = Query(...),
    include_judge_labels: bool = Query(False),
    session: AsyncSession = Depends(get_session),
):
    dataset = await session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    stmt = (
        select(PromptSample)
        .where(PromptSample.dataset_id == dataset_id)
        .options(selectinload(PromptSample.annotations))
        .order_by(PromptSample.active_learning_priority.desc())
    )
    samples = (await session.execute(stmt)).scalars().all()

    pairs = build_preference_pairs(
        ((s, s.annotations) for s in samples),
        include_judge_labels=include_judge_labels,
    )

    sources: dict[str, int] = {}
    examples: list[dict] = []
    for p in pairs:
        sources[p.source] = sources.get(p.source, 0) + 1
        if len(examples) < 10:
            examples.append({
                "sample_id": p.sample_id,
                "prompt": p.prompt[:160],
                "chosen": p.chosen[:160],
                "rejected": p.rejected[:160],
                "source": p.source,
            })

    return PairPreview(
        dataset_id=dataset_id,
        num_pairs=len(pairs),
        sources=sources,
        examples=examples,
    )


@router.post("/runs", response_model=TrainingRunOut, status_code=status.HTTP_202_ACCEPTED)
async def launch_training_run(
    payload: TrainingRunCreate,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    dataset = await session.get(Dataset, payload.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    settings = get_settings()
    base_model = payload.base_model or settings.base_model

    config_dict = payload.model_dump(exclude={"dataset_id", "base_model"})
    run = TrainingRun(
        dataset_id=payload.dataset_id,
        base_model=base_model,
        config=config_dict,
        status=RunStatus.QUEUED,
        metrics={"phase": "queued"},
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    dispatch_training(background, str(run.id))
    return run


@router.get("/runs", response_model=list[TrainingRunOut])
async def list_training_runs(
    dataset_id: Optional[uuid.UUID] = None,
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(TrainingRun)
    if dataset_id:
        stmt = stmt.where(TrainingRun.dataset_id == dataset_id)
    stmt = stmt.order_by(TrainingRun.created_at.desc()).limit(limit)
    res = await session.execute(stmt)
    return res.scalars().all()


@router.get("/runs/{run_id}", response_model=TrainingRunOut)
async def get_training_run(run_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    run = await session.get(TrainingRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run
