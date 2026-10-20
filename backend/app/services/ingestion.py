"""Prompt ingestion (embedding, semantic dedupe, candidate generation, scoring) and status rules."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.domain import Annotation, Dataset, PromptSample, SampleStatus
from app.schemas.payloads import PromptIngestItem
from app.services.active_learning import PriorityWeights, cosine_similarity, priority_score, semantic_divergence
from app.services.dpo_trainer import majority_human_choice
from app.services.inference_engine import generate_pair, get_embedder, get_generator


class IngestionError(ValueError):
    pass


async def _is_duplicate(session: AsyncSession, dataset_id, vec: list[float], seen: list[list[float]]) -> bool:
    threshold = get_settings().dedupe_threshold
    if any(cosine_similarity(vec, other) >= threshold for other in seen):
        return True
    if session.bind.dialect.name == "postgresql":
        # pgvector ANN lookup: cosine distance = 1 - cosine similarity
        stmt = (
            select(PromptSample.id)
            .where(PromptSample.dataset_id == dataset_id)
            .where(PromptSample.embedding.cosine_distance(vec) <= 1 - threshold)
            .limit(1)
        )
        return (await session.execute(stmt)).first() is not None
    rows = (await session.execute(
        select(PromptSample.embedding).where(PromptSample.dataset_id == dataset_id)
    )).scalars().all()
    return any(r is not None and cosine_similarity(vec, r) >= threshold for r in rows)


async def ingest_prompts(
    session: AsyncSession, dataset: Dataset, items: Sequence[PromptIngestItem], skip_duplicates: bool = True
) -> tuple[list[PromptSample], list[str]]:
    settings = get_settings()
    embedder = get_embedder()
    weights = PriorityWeights(settings.priority_entropy_weight, settings.priority_divergence_weight)
    created: list[PromptSample] = []
    skipped: list[str] = []
    seen: list[list[float]] = []

    for i, item in enumerate(items):
        prompt_vec = embedder.embed([item.user_prompt])[0]
        if skip_duplicates and await _is_duplicate(session, dataset.id, prompt_vec, seen):
            skipped.append(item.user_prompt[:120])
            continue
        seen.append(prompt_vec)

        a_text, a_model, a_ent = item.candidate_a, item.candidate_a_model, item.candidate_a_logprob_entropy
        b_text, b_model, b_ent = item.candidate_b, item.candidate_b_model, item.candidate_b_logprob_entropy
        if not (a_text and b_text):
            generator = get_generator()
            if generator is None:
                raise IngestionError(
                    f"Item {i} has no candidates and GENERATOR_BACKEND=none. Supply candidate_a/candidate_b "
                    "or set GENERATOR_BACKEND=local|litellm."
                )
            ca, cb = await asyncio.to_thread(generate_pair, generator, item.system_prompt, item.user_prompt)
            a_text, a_model, a_ent = ca.text, ca.model, ca.entropy
            b_text, b_model, b_ent = cb.text, cb.model, cb.entropy

        emb_a, emb_b = embedder.embed([a_text, b_text])
        divergence = round(semantic_divergence(emb_a, emb_b), 6)
        sample = PromptSample(
            dataset_id=dataset.id,
            category=item.category,
            system_prompt=item.system_prompt,
            user_prompt=item.user_prompt,
            embedding=prompt_vec,
            candidate_a=a_text,
            candidate_a_model=a_model,
            candidate_a_logprob_entropy=a_ent,
            candidate_b=b_text,
            candidate_b_model=b_model,
            candidate_b_logprob_entropy=b_ent,
            semantic_divergence_score=divergence,
            active_learning_priority=priority_score(a_ent, b_ent, divergence, weights),
            status=SampleStatus.PENDING,
            meta=item.meta,
        )
        session.add(sample)
        created.append(sample)

    await session.commit()
    return created, skipped


def derive_status(current: SampleStatus, annotations: Sequence[Annotation]) -> SampleStatus:
    """PENDING until a human labels it; DISPUTED if humans split or the judge disagrees with them."""
    if current == SampleStatus.EXPORTED:
        return current
    humans = [a for a in annotations if a.annotator_type.value == "HUMAN"]
    if not humans:
        return SampleStatus.PENDING
    human_votes = Counter(a.preference_choice.value for a in humans)
    majority = majority_human_choice(humans)
    if majority is None or len(human_votes) > 1:
        return SampleStatus.DISPUTED
    judges = [a for a in annotations if a.annotator_type.value == "LLM_JUDGE"]
    if judges and judges[-1].preference_choice.value != majority:
        return SampleStatus.DISPUTED
    return SampleStatus.ANNOTATED


async def refresh_sample_status(session: AsyncSession, sample: PromptSample) -> SampleStatus:
    anns = (await session.execute(
        select(Annotation).where(Annotation.sample_id == sample.id).order_by(Annotation.created_at)
    )).scalars().all()
    sample.status = derive_status(sample.status, anns)
    return sample.status
