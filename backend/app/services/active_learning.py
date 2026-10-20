"""
Active-learning queue prioritisation.

A sample is worth a human's time when the model is *uncertain* about it. We measure that with
two complementary signals:

1. Token log-prob entropy — mean Shannon entropy of the per-token top-k distributions while
   generating each candidate. High entropy => the model was guessing.
2. Semantic divergence — 1 - cosine(embedding(A), embedding(B)). Two samples of the same prompt
   that say very different things => the model has no stable answer.

The combined priority is in [0, 1]. Items above `human_review_threshold` go to humans; items
below it can be auto-labelled by the (calibrated) LLM judge.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

T = TypeVar("T")

# Entropy of a uniform distribution over the top-5 alternatives; used to scale entropy into [0, 1].
DEFAULT_TOP_K = 5
MAX_TOPK_ENTROPY = math.log(DEFAULT_TOP_K)


@dataclass(frozen=True)
class PriorityWeights:
    entropy: float = 0.5
    divergence: float = 0.5


def token_entropy(top_logprobs: Sequence[Sequence[float]]) -> float:
    """Mean per-token entropy (nats) of renormalised top-k log-prob distributions."""
    entropies: list[float] = []
    for dist in top_logprobs:
        if len(dist) == 0:
            continue
        lp = np.asarray(dist, dtype=float)
        p = np.exp(lp - lp.max())
        p /= p.sum()
        entropies.append(float(-(p * np.log(p + 1e-12)).sum()))
    return float(np.mean(entropies)) if entropies else 0.0


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def semantic_divergence(emb_a: Sequence[float], emb_b: Sequence[float]) -> float:
    """1 - cosine similarity, clipped to [0, 1]."""
    return float(np.clip(1.0 - cosine_similarity(emb_a, emb_b), 0.0, 1.0))


def priority_score(
    entropy_a: float | None,
    entropy_b: float | None,
    divergence: float | None,
    weights: PriorityWeights = PriorityWeights(),
    entropy_scale: float = MAX_TOPK_ENTROPY,
) -> float:
    """Combine uncertainty signals into a [0, 1] priority.

    If log-probs are unavailable (e.g. candidates were imported as plain text), the score falls back
    to semantic divergence alone rather than inventing an entropy value.
    """
    div = float(np.clip(divergence or 0.0, 0.0, 1.0))
    entropies = [e for e in (entropy_a, entropy_b) if e is not None]
    if not entropies:
        return round(div, 6)
    ent = float(np.clip(np.mean(entropies) / entropy_scale, 0.0, 1.0))
    total = weights.entropy + weights.divergence
    if total <= 0:
        raise ValueError("Priority weights must sum to a positive number")
    return round((weights.entropy * ent + weights.divergence * div) / total, 6)


def rank_queue(items: Sequence[T], key: Callable[[T], float]) -> list[T]:
    """Highest-priority first (stable for ties)."""
    return sorted(items, key=key, reverse=True)


def route_for_review(priorities: Sequence[float], threshold: float) -> dict[str, float | int]:
    """Split a queue into human-review vs judge-auto-label buckets.

    `auto_label_fraction` is the share of items that do *not* need a human — i.e. the realised
    labelling-cost saving at this threshold.
    """
    n = len(priorities)
    human = sum(1 for p in priorities if p >= threshold)
    return {
        "total": n,
        "human_review": human,
        "auto_label": n - human,
        "auto_label_fraction": round((n - human) / n, 4) if n else 0.0,
    }
