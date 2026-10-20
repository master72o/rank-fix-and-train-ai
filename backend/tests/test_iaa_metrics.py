"""Unit tests for Inter-Annotator Agreement (Cohen's Kappa), confusion matrices, and bootstrap CIs."""

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from app.services.iaa_metrics import (
    PREFERENCE_LABELS,
    bootstrap_kappa_ci,
    cohen_kappa,
    compute_agreement,
    confusion_matrix,
    interpret_kappa,
    kappa_from_matrix,
    percent_agreement,
    position_consistency_rate,
)


def test_confusion_matrix_shape_and_content():
    r1 = ["CANDIDATE_A", "CANDIDATE_B", "TIE", "BOTH_BAD"]
    r2 = ["CANDIDATE_A", "CANDIDATE_A", "TIE", "BOTH_BAD"]
    labels = PREFERENCE_LABELS

    cm = confusion_matrix(r1, r2, labels)
    assert cm.shape == (4, 4)
    # Total pairs
    assert cm.sum() == 4
    # r1[0]=A, r2[0]=A -> cm[0, 0] == 1
    assert cm[0, 0] == 1
    # r1[1]=B, r2[1]=A -> cm[1, 0] == 1
    assert cm[1, 0] == 1


def test_cohen_kappa_matches_scikit_learn():
    r1 = [
        "CANDIDATE_A", "CANDIDATE_A", "CANDIDATE_B", "CANDIDATE_B",
        "TIE", "BOTH_BAD", "CANDIDATE_A", "CANDIDATE_B", "TIE", "CANDIDATE_A",
        "CANDIDATE_B", "BOTH_BAD", "CANDIDATE_A", "TIE", "CANDIDATE_B",
    ]
    r2 = [
        "CANDIDATE_A", "CANDIDATE_B", "CANDIDATE_B", "CANDIDATE_B",
        "TIE", "BOTH_BAD", "CANDIDATE_A", "TIE", "TIE", "CANDIDATE_A",
        "CANDIDATE_B", "CANDIDATE_B", "CANDIDATE_A", "TIE", "CANDIDATE_A",
    ]
    labels = PREFERENCE_LABELS

    custom_k = cohen_kappa(r1, r2, labels=labels)
    sklearn_k = cohen_kappa_score(r1, r2, labels=labels)

    assert pytest.approx(custom_k, abs=1e-4) == sklearn_k
    assert percent_agreement(r1, r2) > 0.60


def test_perfect_agreement():
    r1 = ["CANDIDATE_A", "CANDIDATE_B", "TIE", "BOTH_BAD"] * 5
    r2 = list(r1)
    k = cohen_kappa(r1, r2, labels=PREFERENCE_LABELS)
    assert pytest.approx(k, abs=1e-4) == 1.0
    assert interpret_kappa(k) == "almost perfect"


def test_bootstrap_confidence_interval():
    r1 = ["CANDIDATE_A", "CANDIDATE_B", "TIE", "CANDIDATE_A", "CANDIDATE_B"] * 4
    r2 = ["CANDIDATE_A", "CANDIDATE_A", "TIE", "CANDIDATE_A", "CANDIDATE_B"] * 4
    lo, hi = bootstrap_kappa_ci(r1, r2, labels=PREFERENCE_LABELS, n_boot=500, seed=42)

    assert lo <= hi
    assert 0.3 <= lo <= 1.0
    assert 0.5 <= hi <= 1.0


def test_position_consistency_rate():
    flags = [True, True, False, True, None]
    rate = position_consistency_rate(flags)
    # 3 True out of 4 non-None -> 0.75
    assert rate == 0.75

    assert position_consistency_rate([None, None]) is None


def test_compute_agreement_full_report():
    human = ["CANDIDATE_A", "CANDIDATE_B", "TIE", "CANDIDATE_A"] * 3
    judge = ["CANDIDATE_A", "CANDIDATE_B", "TIE", "CANDIDATE_B"] * 3

    res = compute_agreement(human, judge, labels=PREFERENCE_LABELS, n_boot=200)
    assert res.n_pairs == 12
    assert 0.0 <= res.cohen_kappa <= 1.0
    assert res.percent_agreement == 0.75
    assert len(res.confusion_matrix) == 4
