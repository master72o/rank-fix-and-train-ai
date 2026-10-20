"""
Post-training evaluation: head-to-head win rate of the DPO-tuned adapter vs. the base model on
held-out prompts, judged with the same position-swapped judge used for annotation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from app.services.llm_judge import DEFAULT_RUBRIC, Judge, evaluate_with_position_swap

GenerateFn = Callable[[str | None, str], str]


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class WinRateReport:
    n: int
    wins: int
    losses: int
    ties: int
    win_rate: float  # (wins + 0.5 * ties) / n
    ci_low: float
    ci_high: float
    position_inconsistent: int
    judge: str
    examples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_win_rate(
    prompts: Sequence[tuple[str | None, str]],
    baseline_fn: GenerateFn,
    candidate_fn: GenerateFn,
    judge: Judge,
    rubric: dict[str, str] | None = None,
    max_examples: int = 5,
) -> WinRateReport:
    wins = losses = ties = inconsistent = 0
    examples: list[dict] = []
    for system_prompt, user_prompt in prompts:
        tuned, base = candidate_fn(system_prompt, user_prompt), baseline_fn(system_prompt, user_prompt)
        verdict = evaluate_with_position_swap(judge, user_prompt, tuned, base, rubric or DEFAULT_RUBRIC)
        inconsistent += 0 if verdict.position_consistent else 1
        if verdict.preference == "CANDIDATE_A":
            wins += 1
            outcome = "win"
        elif verdict.preference == "CANDIDATE_B":
            losses += 1
            outcome = "loss"
        else:
            ties += 1
            outcome = "tie"
        if len(examples) < max_examples:
            examples.append({"prompt": user_prompt, "tuned": tuned, "base": base, "outcome": outcome})
    n = len(prompts)
    score = wins + 0.5 * ties
    lo, hi = wilson_interval(score, n)
    return WinRateReport(
        n=n, wins=wins, losses=losses, ties=ties,
        win_rate=round(score / n, 4) if n else 0.0,
        ci_low=round(lo, 4), ci_high=round(hi, 4),
        position_inconsistent=inconsistent, judge=judge.name, examples=examples,
    )


def build_hf_generators(base_model: str, adapter_path: str, max_new_tokens: int = 256) -> tuple[GenerateFn, GenerateFn]:
    """Greedy generation from the base model and from base+adapter, sharing one set of weights."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, adapter_path).to(device)
    model.eval()

    def _generate(system_prompt: str | None, user_prompt: str) -> str:
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def baseline(system_prompt: str | None, user_prompt: str) -> str:
        with model.disable_adapter():
            return _generate(system_prompt, user_prompt)

    def tuned(system_prompt: str | None, user_prompt: str) -> str:
        return _generate(system_prompt, user_prompt)

    return baseline, tuned
