"""API routes for human annotations, rewrites, and rubric scoring."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.models.domain import Annotation, AnnotatorType, PromptSample
from app.schemas.payloads import (
    AnnotationCreate,
    AnnotationCreated,
    AnnotationOut,
)
from app.services.ingestion import refresh_sample_status

router = APIRouter(prefix="/annotations", tags=["annotations"])


@router.post("", response_model=AnnotationCreated, status_code=status.HTTP_201_CREATED)
async def submit_annotation(payload: AnnotationCreate, session: AsyncSession = Depends(get_session)):
    sample = await session.get(PromptSample, payload.sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Prompt sample not found")

    ann = Annotation(
        sample_id=payload.sample_id,
        annotator_id=payload.annotator_id,
        annotator_type=AnnotatorType.HUMAN,
        preference_choice=payload.preference_choice,
        rewritten_chosen_response=payload.rewritten_chosen_response,
        rubric_scores=payload.rubric_scores,
        error_flags=[f.value for f in payload.error_flags],
        critique_rationale=payload.critique_rationale,
        time_spent_ms=payload.time_spent_ms,
    )
    session.add(ann)
    await session.flush()

    new_status = await refresh_sample_status(session, sample)
    await session.commit()
    await session.refresh(ann)

    return AnnotationCreated(
        annotation=AnnotationOut.model_validate(ann),
        sample_status=new_status,
    )


@router.get("", response_model=list[AnnotationOut])
async def list_annotations(
    sample_id: Optional[uuid.UUID] = None,
    dataset_id: Optional[uuid.UUID] = None,
    limit: int = Query(50, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Annotation)
    if sample_id:
        stmt = stmt.where(Annotation.sample_id == sample_id)
    if dataset_id:
        stmt = stmt.join(PromptSample, Annotation.sample_id == PromptSample.id).where(PromptSample.dataset_id == dataset_id)
    stmt = stmt.order_by(Annotation.created_at.desc()).limit(limit)
    res = await session.execute(stmt)
    return res.scalars().all()


@router.get("/{annotation_id}", response_model=AnnotationOut)
async def get_annotation(annotation_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    ann = await session.get(Annotation, annotation_id)
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return ann
