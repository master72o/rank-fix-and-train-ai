"""API routes for LLM-as-a-Judge execution and Inter-Annotator Agreement (Cohen's Kappa) reporting."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.database import get_session
from app.models.domain import Annotation, AnnotatorType, Dataset, PromptSample
from app.schemas.payloads import (
    AgreementReport,
    Disagreement,
    JobAccepted,
    JudgeRunRequest,
)
from app.services.iaa_metrics import (
    PREFERENCE_LABELS,
    compute_agreement,
    position_consistency_rate,
)
from app.services.llm_judge import get_judge
from app.workers.tasks import dispatch_judge

router = APIRouter(prefix="/judge", tags=["judge"])


@router.post("/run", response_model=JobAccepted)
async def run_judge_evaluation(
    payload: JudgeRunRequest,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    dataset = await session.get(Dataset, payload.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    stmt = select(PromptSample.id).where(PromptSample.dataset_id == payload.dataset_id)
    if payload.sample_ids:
        stmt = stmt.where(PromptSample.id.in_(payload.sample_ids))
    if payload.only_human_annotated:
        stmt = stmt.join(Annotation, Annotation.sample_id == PromptSample.id).where(
            Annotation.annotator_type == AnnotatorType.HUMAN
        ).distinct()

    stmt = stmt.limit(payload.limit)
    res = await session.execute(stmt)
    sample_ids = [str(r) for r in res.scalars().all()]

    if not sample_ids:
        raise HTTPException(status_code=400, detail="No matching prompt samples found for judge evaluation.")

    mode = dispatch_judge(background, sample_ids)
    return JobAccepted(
        job="judge_samples",
        queued=len(sample_ids),
        mode=mode,
    )


@router.get("/agreement", response_model=AgreementReport)
async def get_agreement_report(
    dataset_id: uuid.UUID = Query(...),
    judge_name: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    dataset = await session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    target_judge = judge_name or get_judge().name

    stmt = (
        select(PromptSample)
        .where(PromptSample.dataset_id == dataset_id)
        .options(selectinload(PromptSample.annotations))
    )
    res = await session.execute(stmt)
    samples = res.scalars().all()

    human_choices: list[str] = []
    judge_choices: list[str] = []
    disagreements: list[Disagreement] = []
    consistency_flags: list[bool | None] = []

    judge_dist: dict[str, int] = {k: 0 for k in PREFERENCE_LABELS}
    human_dist: dict[str, int] = {k: 0 for k in PREFERENCE_LABELS}

    gold_matches = 0
    gold_total = 0

    total_judge_verdicts = 0

    for s in samples:
        # Check if sample has gold reference in meta
        gold_pref = s.meta.get("gold_preference") if isinstance(s.meta, dict) else None

        # Filter annotations
        judge_ann = next(
            (a for a in s.annotations if a.annotator_type == AnnotatorType.LLM_JUDGE and a.annotator_id == target_judge),
            None,
        )
        if not judge_ann:
            # fallback to any judge annotation if specific judge name wasn't specified
            judge_ann = next(
                (a for a in s.annotations if a.annotator_type == AnnotatorType.LLM_JUDGE),
                None,
            )

        if judge_ann:
            total_judge_verdicts += 1
            j_choice = judge_ann.preference_choice.value
            judge_dist[j_choice] = judge_dist.get(j_choice, 0) + 1
            consistency_flags.append(judge_ann.position_consistent)

            if gold_pref:
                gold_total += 1
                if j_choice == gold_pref:
                    gold_matches += 1

        human_ann = next((a for a in s.annotations if a.annotator_type == AnnotatorType.HUMAN), None)
        if human_ann:
            h_choice = human_ann.preference_choice.value
            human_dist[h_choice] = human_dist.get(h_choice, 0) + 1

        if human_ann and judge_ann:
            h_choice = human_ann.preference_choice.value
            j_choice = judge_ann.preference_choice.value
            human_choices.append(h_choice)
            judge_choices.append(j_choice)

            if h_choice != j_choice:
                disagreements.append(
                    Disagreement(
                        sample_id=s.id,
                        user_prompt=s.user_prompt,
                        human=h_choice,
                        judge=j_choice,
                        position_consistent=judge_ann.position_consistent,
                    )
                )

    p_consistency = position_consistency_rate(consistency_flags)
    judge_gold_acc = round(gold_matches / gold_total, 4) if gold_total > 0 else None

    if len(human_choices) >= 2:
        try:
            agreement_res = compute_agreement(human_choices, judge_choices, labels=PREFERENCE_LABELS)
            return AgreementReport(
                dataset_id=dataset_id,
                judge_name=target_judge,
                n_pairs=agreement_res.n_pairs,
                cohen_kappa=agreement_res.cohen_kappa,
                kappa_ci_low=agreement_res.kappa_ci_low,
                kappa_ci_high=agreement_res.kappa_ci_high,
                kappa_se=agreement_res.kappa_se,
                kappa_interpretation=agreement_res.kappa_interpretation,
                percent_agreement=agreement_res.percent_agreement,
                labels=agreement_res.labels,
                confusion_matrix=agreement_res.confusion_matrix,
                position_consistency_rate=p_consistency,
                judge_verdicts=total_judge_verdicts,
                judge_label_distribution=judge_dist,
                human_label_distribution=human_dist,
                judge_gold_accuracy=judge_gold_acc,
                gold_items=gold_total,
                disagreements=disagreements[:20],
                message=None,
            )
        except Exception as e:
            return AgreementReport(
                dataset_id=dataset_id,
                judge_name=target_judge,
                n_pairs=len(human_choices),
                labels=PREFERENCE_LABELS,
                position_consistency_rate=p_consistency,
                judge_verdicts=total_judge_verdicts,
                judge_label_distribution=judge_dist,
                human_label_distribution=human_dist,
                judge_gold_accuracy=judge_gold_acc,
                gold_items=gold_total,
                disagreements=disagreements[:20],
                message=f"Agreement computation note: {e}",
            )

    return AgreementReport(
        dataset_id=dataset_id,
        judge_name=target_judge,
        n_pairs=len(human_choices),
        labels=PREFERENCE_LABELS,
        position_consistency_rate=p_consistency,
        judge_verdicts=total_judge_verdicts,
        judge_label_distribution=judge_dist,
        human_label_distribution=human_dist,
        judge_gold_accuracy=judge_gold_acc,
        gold_items=gold_total,
        disagreements=disagreements[:20],
        message="Need at least 2 overlapping human and judge annotations to calculate Cohen's Kappa.",
    )
