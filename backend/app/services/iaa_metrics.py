"""
Inter-annotator agreement between human reviewers and the LLM judge.

* Cohen's kappa from the confusion matrix (validated against scikit-learn in tests)
* Percentile-bootstrap confidence interval + analytic standard error (statsmodels)
* Landis & Koch interpretation bands
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

PREFERENCE_LABELS = ["CANDIDATE_A", "CANDIDATE_B", "TIE", "BOTH_BAD"]


def confusion_matrix(r1: Sequence[str], r2: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    if len(r1) != len(r2):
        raise ValueError("Rater label sequences must have equal length")
    index = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b in zip(r1, r2, strict=True):
        if a not in index or b not in index:
            raise ValueError(f"Unknown label pair ({a!r}, {b!r})")
        cm[index[a], index[b]] += 1
    return cm


def kappa_from_matrix(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        raise ValueError("Cannot compute kappa on zero items")
    po = np.trace(cm) / n
    pe = float((cm.sum(axis=1) * cm.sum(axis=0)).sum()) / (n * n)
    if np.isclose(pe, 1.0):
        # Both raters used a single identical category: agreement is perfect but uninformative.
        return 1.0 if np.isclose(po, 1.0) else 0.0
    return float((po - pe) / (1.0 - pe))


def cohen_kappa(r1: Sequence[str], r2: Sequence[str], labels: Sequence[str] | None = None) -> float:
    labels = list(labels) if labels else sorted(set(r1) | set(r2))
    return kappa_from_matrix(confusion_matrix(r1, r2, labels))


def percent_agreement(r1: Sequence[str], r2: Sequence[str]) -> float:
    if not r1:
        raise ValueError("Cannot compute agreement on zero items")
    return float(np.mean([a == b for a, b in zip(r1, r2, strict=True)]))


def bootstrap_kappa_ci(
    r1: Sequence[str],
    r2: Sequence[str],
    labels: Sequence[str] | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    labels = list(labels) if labels else sorted(set(r1) | set(r2))
    a, b = np.asarray(r1), np.asarray(r2)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        stats[i] = kappa_from_matrix(confusion_matrix(a[idx], b[idx], labels))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def analytic_kappa_se(cm: np.ndarray) -> float | None:
    """Asymptotic standard error of kappa via statsmodels (None if unavailable/degenerate)."""
    try:
        from statsmodels.stats.inter_rater import cohens_kappa

        used = cm.sum(axis=0) + cm.sum(axis=1) > 0
        res = cohens_kappa(cm[np.ix_(used, used)])
        se = float(res["std_kappa"])
        return se if np.isfinite(se) else None
    except Exception:
        return None


def interpret_kappa(k: float) -> str:
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def position_consistency_rate(flags: Sequence[bool | None]) -> float | None:
    known = [f for f in flags if f is not None]
    return round(float(np.mean(known)), 4) if known else None


@dataclass
class AgreementResult:
    n_pairs: int
    cohen_kappa: float
    kappa_ci_low: float
    kappa_ci_high: float
    kappa_se: float | None
    kappa_interpretation: str
    percent_agreement: float
    labels: list[str]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_agreement(
    human: Sequence[str],
    judge: Sequence[str],
    labels: Sequence[str] = PREFERENCE_LABELS,
    n_boot: int = 2000,
    seed: int = 0,
) -> AgreementResult:
    cm = confusion_matrix(human, judge, labels)
    k = kappa_from_matrix(cm)
    lo, hi = bootstrap_kappa_ci(human, judge, labels, n_boot=n_boot, seed=seed)
    return AgreementResult(
        n_pairs=len(human),
        cohen_kappa=round(k, 4),
        kappa_ci_low=round(lo, 4),
        kappa_ci_high=round(hi, 4),
        kappa_se=None if (se := analytic_kappa_se(cm)) is None else round(se, 4),
        kappa_interpretation=interpret_kappa(k),
        percent_agreement=round(percent_agreement(human, judge), 4),
        labels=list(labels),
        confusion_matrix=cm.tolist(),
    )
