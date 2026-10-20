"""API routes for datasets and active-learning prompt queues."""

from __future__ import annotations

import uuid
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_session
from app.models.domain import Annotation, AnnotatorType, Dataset, PromptSample, SampleStatus
from app.schemas.payloads import (
    DatasetCreate,
    DatasetOut,
    PromptIngestRequest,
    PromptIngestResponse,
    PromptSampleOut,
    QueueStats,
)
from app.services.active_learning import route_for_review
from app.services.ingestion import IngestionError, ingest_prompts
from app.services.llm_judge import DEFAULT_RUBRIC

router = APIRouter(prefix="/prompts", tags=["prompts"])


@router.post("/datasets", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
async def create_dataset(payload: DatasetCreate, session: AsyncSession = Depends(get_session)):
    existing = await session.execute(select(Dataset).where(Dataset.name == payload.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail=f"Dataset with name '{payload.name}' already exists.")
    dataset = Dataset(
        name=payload.name,
        task_type=payload.task_type,
        rubric_definition=payload.rubric_definition or DEFAULT_RUBRIC,
    )
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)
    return dataset


@router.get("/datasets", response_model=list[DatasetOut])
async def list_datasets(session: AsyncSession = Depends(get_session)):
    stmt = select(Dataset).order_by(Dataset.created_at.desc())
    res = await session.execute(stmt)
    return res.scalars().all()


@router.post("/ingest", response_model=PromptIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_prompts_endpoint(payload: PromptIngestRequest, session: AsyncSession = Depends(get_session)):
    dataset = await session.get(Dataset, payload.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        created, skipped = await ingest_prompts(
            session=session,
            dataset=dataset,
            items=payload.items,
            skip_duplicates=payload.skip_duplicates,
        )
        return PromptIngestResponse(
            created=len(created),
            sample_ids=[s.id for s in created],
            skipped_duplicates=skipped,
        )
    except IngestionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/queue", response_model=list[PromptSampleOut])
async def get_queue(
    dataset_id: Optional[uuid.UUID] = None,
    status_filter: Optional[SampleStatus] = Query(None, alias="status"),
    min_priority: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(PromptSample)
    if dataset_id:
        stmt = stmt.where(PromptSample.dataset_id == dataset_id)
    if status_filter:
        stmt = stmt.where(PromptSample.status == status_filter)
    if min_priority is not None:
        stmt = stmt.where(PromptSample.active_learning_priority >= min_priority)
    stmt = stmt.order_by(PromptSample.active_learning_priority.desc()).offset(offset).limit(limit)
    res = await session.execute(stmt)
    return res.scalars().all()


@router.get("/stats/queue", response_model=QueueStats)
async def get_queue_stats(
    dataset_id: Optional[uuid.UUID] = None,
    session: AsyncSession = Depends(get_session),
):
    settings = get_settings()
    stmt = select(PromptSample)
    if dataset_id:
        stmt = stmt.where(PromptSample.dataset_id == dataset_id)
    res = await session.execute(stmt)
    samples = res.scalars().all()

    total = len(samples)
    status_counts = {}
    priorities = []
    for s in samples:
        st_val = s.status.value if hasattr(s.status, "value") else str(s.status)
        status_counts[st_val] = status_counts.get(st_val, 0) + 1
        priorities.append(float(s.active_learning_priority or 0.0))

    # Annotations count
    ann_stmt = select(Annotation.annotator_type, func.count(Annotation.id)).group_by(Annotation.annotator_type)
    if dataset_id:
        ann_stmt = ann_stmt.join(PromptSample, Annotation.sample_id == PromptSample.id).where(PromptSample.dataset_id == dataset_id)
    ann_res = (await session.execute(ann_stmt)).all()
    human_count = 0
    judge_count = 0
    for atype, cnt in ann_res:
        val = atype.value if hasattr(atype, "value") else str(atype)
        if val == "HUMAN":
            human_count = cnt
        elif val == "LLM_JUDGE":
            judge_count = cnt

    threshold = settings.human_review_threshold
    routing = route_for_review(priorities, threshold)

    # Compute histogram (5 buckets: [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0])
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist_counts, _ = np.histogram(priorities, bins=bins) if priorities else ([0]*5, None)
    histogram = []
    for i in range(len(hist_counts)):
        histogram.append({
            "range": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
            "count": int(hist_counts[i]),
        })

    return QueueStats(
        dataset_id=dataset_id,
        total=total,
        by_status=status_counts,
        human_annotations=human_count,
        judge_annotations=judge_count,
        review_threshold=threshold,
        routing=routing,
        priority_histogram=histogram,
    )


@router.get("/{sample_id}", response_model=PromptSampleOut)
async def get_sample(sample_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    sample = await session.get(PromptSample, sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Prompt sample not found")
    return sample
