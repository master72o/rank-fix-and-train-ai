"""Unit tests for active learning uncertainty, entropy, and queue prioritization."""

import math
import pytest
from app.services.active_learning import (
    token_entropy,
    cosine_similarity,
    semantic_divergence,
    length_ratio_divergence,
    priority_score,
    route_for_review,
    calibrate_threshold_for_budget,
    rank_queue,
    PriorityWeights,
)


def test_token_entropy_uniform_vs_peaked():
    # Uniform top-5: p = 0.2, entropy = -5 * (0.2 * ln(0.2)) = ln(5) ~= 1.609
    uniform_lps = [[math.log(0.2)] * 5 for _ in range(4)]
    ent_uniform = token_entropy(uniform_lps)
    assert 1.55 <= ent_uniform <= 1.65

    # Peaked (almost 1.0 on top-1): entropy should be near 0
    peaked_lps = [[0.0, -10.0, -10.0, -10.0, -10.0] for _ in range(4)]
    ent_peaked = token_entropy(peaked_lps)
    assert ent_peaked < 0.05


def test_cosine_similarity_and_divergence():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0
    assert pytest.approx(semantic_divergence(v1, v2)) == 0.0

    assert pytest.approx(cosine_similarity(v1, v3)) == 0.0
    assert pytest.approx(semantic_divergence(v1, v3)) == 1.0

    # Zero vector safety
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_priority_score_calculation():
    # High entropy and high divergence -> high priority
    weights = PriorityWeights(entropy=0.5, divergence=0.5)
    score_high = priority_score(entropy_a=1.5, entropy_b=1.5, divergence=0.9, weights=weights)
    assert 0.8 <= score_high <= 1.0

    # Low entropy and low divergence -> low priority
    score_low = priority_score(entropy_a=0.05, entropy_b=0.05, divergence=0.05, weights=weights)
    assert score_low < 0.2

    # None entropy fallback (divergence-only)
    score_div_only = priority_score(entropy_a=None, entropy_b=None, divergence=0.75, weights=weights)
    assert score_div_only == 0.75


def test_route_for_review_savings():
    priorities = [0.9, 0.8, 0.5, 0.4, 0.2, 0.1, 0.05]
    routing = route_for_review(priorities, threshold=0.35)

    assert routing["total"] == 7
    # 0.9, 0.8, 0.5, 0.4 are >= 0.35 -> 4 human
    assert routing["human_review"] == 4
    # 0.2, 0.1, 0.05 are < 0.35 -> 3 auto-label
    assert routing["auto_label"] == 3
    assert pytest.approx(routing["auto_label_fraction"], rel=1e-2) == 3 / 7


def test_rank_queue():
    items = [
        {"id": "a", "priority": 0.2},
        {"id": "b", "priority": 0.8},
        {"id": "c", "priority": 0.5},
    ]
    ranked = rank_queue(items, key=lambda x: x["priority"])
    assert [x["id"] for x in ranked] == ["b", "c", "a"]


def test_length_ratio_divergence_and_three_signal_priority():
    assert length_ratio_divergence(100, 100) == 0.0
    # e^1 ratio -> saturated at 1.0
    assert length_ratio_divergence(300, 100) == 1.0
    mid_div = length_ratio_divergence(150, 100)
    assert 0.35 <= mid_div <= 0.45

    weights = PriorityWeights(entropy=0.4, divergence=0.4, length=0.2)
    score = priority_score(
        entropy_a=1.609,
        entropy_b=1.609,
        divergence=0.5,
        weights=weights,
        length_div=1.0,
    )
    # 0.4*1.0 + 0.4*0.5 + 0.2*1.0 = 0.8
    assert pytest.approx(score, abs=1e-2) == 0.8


def test_calibrate_threshold_for_budget():
    priorities = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    res = calibrate_threshold_for_budget(priorities, target_human_fraction=0.3)
    assert res["total"] == 10
    assert res["human_review"] == 3
    assert res["auto_label"] == 7
    assert res["auto_label_fraction"] == 0.7
    assert res["threshold"] == 0.75
