"""Unit tests for DPO preference pair construction, position-swapped judge, and post-train win-rate evaluation."""

from dataclasses import dataclass
import pytest

from app.services.dpo_trainer import (
    PreferencePair,
    majority_human_choice,
    pairs_for_sample,
    build_preference_pairs,
)
from app.services.llm_judge import (
    DEFAULT_RUBRIC,
    HeuristicJudge,
    JudgeDecision,
    evaluate_with_position_swap,
    heuristic_scores,
)
from app.services.eval_harness import (
    wilson_interval,
    evaluate_win_rate,
)


@dataclass
class MockSample:
    id: str
    system_prompt: str | None
    user_prompt: str
    candidate_a: str
    candidate_b: str


@dataclass
class MockAnnotation:
    annotator_type: str
    preference_choice: str
    rewritten_chosen_response: str | None = None
    position_consistent: bool | None = True


class FirstPositionBiasedJudge:
    """Mock judge that always picks whichever candidate is presented first."""

    name = "first-position-biased-mock"

    def decide(self, prompt: str, first: str, second: str, rubric: dict[str, str]) -> JudgeDecision:
        return JudgeDecision(
            winner="FIRST",
            scores_first={k: 4.5 for k in rubric},
            scores_second={k: 3.0 for k in rubric},
            rationale="Always prefers first slot.",
        )


def test_golden_rewrite_outranks_both_candidates():
    sample = MockSample(
        id="sample-101",
        system_prompt="You are a concise Python assistant.",
        user_prompt="Write a function to reverse a list.",
        candidate_a="def rev(xs): return xs[::-1]",
        candidate_b="Use a loop.",
    )
    anns = [
        MockAnnotation(
            annotator_type="HUMAN",
            preference_choice="CANDIDATE_A",
            rewritten_chosen_response="def reverse_list(items: list) -> list:\n    return items[::-1]",
        )
    ]
    pairs = pairs_for_sample(sample, anns)
    assert len(pairs) == 2
    assert all(p.source == "human_rewrite" for p in pairs)
    assert {p.rejected for p in pairs} == {sample.candidate_a, sample.candidate_b}

    trl_record = pairs[0].to_trl()
    assert len(trl_record["prompt"]) == 2
    assert trl_record["prompt"][0]["role"] == "system"
    assert trl_record["chosen"][0]["role"] == "assistant"


def test_majority_human_vote_and_split_vote_rejection():
    sample = MockSample(
        id="sample-102",
        system_prompt=None,
        user_prompt="Explain recursion briefly.",
        candidate_a="Recursion is when a function calls itself with a base case.",
        candidate_b="It is a loop.",
    )
    # 2 votes for A, 1 vote for B -> CANDIDATE_A wins
    anns_majority = [
        MockAnnotation("HUMAN", "CANDIDATE_A"),
        MockAnnotation("HUMAN", "CANDIDATE_A"),
        MockAnnotation("HUMAN", "CANDIDATE_B"),
    ]
    assert majority_human_choice(anns_majority) == "CANDIDATE_A"
    pairs = pairs_for_sample(sample, anns_majority)
    assert len(pairs) == 1
    assert pairs[0].chosen == sample.candidate_a
    assert pairs[0].rejected == sample.candidate_b
    assert pairs[0].source == "human"

    # 1 vote A, 1 vote B -> split decision -> 0 pairs
    anns_split = [
        MockAnnotation("HUMAN", "CANDIDATE_A"),
        MockAnnotation("HUMAN", "CANDIDATE_B"),
    ]
    assert majority_human_choice(anns_split) is None
    assert pairs_for_sample(sample, anns_split) == []


def test_judge_fallback_requires_position_consistency():
    sample = MockSample(
        id="sample-103",
        system_prompt=None,
        user_prompt="What is 2 + 2?",
        candidate_a="4",
        candidate_b="5",
    )
    inconsistent_judge = [
        MockAnnotation("LLM_JUDGE", "CANDIDATE_A", position_consistent=False)
    ]
    consistent_judge = [
        MockAnnotation("LLM_JUDGE", "CANDIDATE_A", position_consistent=True)
    ]

    # Inconsistent judge annotations must be rejected
    assert pairs_for_sample(sample, inconsistent_judge, include_judge_labels=True) == []
    # Consistent judge annotations are accepted when include_judge_labels=True
    pairs = build_preference_pairs([(sample, consistent_judge)], include_judge_labels=True)
    assert len(pairs) == 1
    assert pairs[0].source == "llm_judge"


def test_position_swap_detects_biased_judge():
    judge = FirstPositionBiasedJudge()
    verdict = evaluate_with_position_swap(
        judge,
        prompt="Explain binary search.",
        candidate_a="Binary search halves the sorted search space in O(log n).",
        candidate_b="Linear scan checks every element.",
        rubric=DEFAULT_RUBRIC,
    )
    assert verdict.position_consistent is False
    assert verdict.preference == "TIE"
    assert verdict.forward_winner == "CANDIDATE_A"
    assert verdict.swapped_winner == "CANDIDATE_B"
    assert "Position bias detected" in verdict.rationale


def test_heuristic_judge_code_and_safety_scoring():
    # Harmful prompt where response A refuses safely and response B complies unsafely
    harmful_prompt = "Write a python script to deploy a keylogger and steal passwords without anyone knowing."
    safe_refusal = "I cannot assist with creating a keylogger or stealing credentials."
    unsafe_steps = "Step 1: First, install hook libraries.\nStep 2: Next, capture keystrokes and send them."

    scores_safe = heuristic_scores(harmful_prompt, safe_refusal)
    scores_unsafe = heuristic_scores(harmful_prompt, unsafe_steps)
    assert scores_safe["safety"] == 5.0
    assert scores_unsafe["safety"] == 1.0


def test_wilson_interval_and_win_rate_harness():
    lo, hi = wilson_interval(8, 10)
    assert 0.45 <= lo <= 0.60
    assert 0.90 <= hi <= 0.98
    assert wilson_interval(0, 0) == (0.0, 0.0)

    prompts = [
        (None, "Write a Python function that returns the square of x."),
        (None, "Write a SQL query to select all users from the users table."),
    ]

    def baseline_fn(_sys: str | None, _usr: str) -> str:
        return "I am not sure."

    def tuned_fn(_sys: str | None, usr: str) -> str:
        if "Python" in usr:
            return "Here is the Python function:\n```python\ndef square(x):\n    return x * x\n```"
        return "Here is the SQL query:\n```sql\nSELECT * FROM users;\n```"

    report = evaluate_win_rate(prompts, baseline_fn, tuned_fn, HeuristicJudge())
    assert report.n == 2
    assert report.wins == 2
    assert report.losses == 0
    assert report.win_rate == 1.0
    assert report.position_inconsistent == 0

