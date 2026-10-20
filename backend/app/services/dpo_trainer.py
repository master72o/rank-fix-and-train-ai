"""
Preference-pair construction and QLoRA/LoRA Direct Preference Optimization with HuggingFace TRL.

Pair construction rules (humans always outrank the judge):
* A human rewrite becomes the *chosen* response and is paired against every candidate it replaced.
* Otherwise the human majority vote decides (CANDIDATE_A / CANDIDATE_B). TIE, BOTH_BAD, or a split
  vote produce no pair.
* Judge labels are only used when `include_judge_labels=True`, no human labelled the sample, and the
  verdict survived the position swap.
"""

from __future__ import annotations

import importlib.util
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


class AnnotationLike(Protocol):
    annotator_type: Any
    preference_choice: Any
    rewritten_chosen_response: str | None
    position_consistent: bool | None


class SampleLike(Protocol):
    id: Any
    system_prompt: str | None
    user_prompt: str
    candidate_a: str
    candidate_b: str


@dataclass
class PreferencePair:
    sample_id: str
    system_prompt: str | None
    prompt: str
    chosen: str
    rejected: str
    source: str  # human | human_rewrite | llm_judge

    def to_trl(self) -> dict[str, list[dict[str, str]]]:
        """TRL conversational preference format."""
        prompt = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        prompt.append({"role": "user", "content": self.prompt})
        return {
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": self.chosen}],
            "rejected": [{"role": "assistant", "content": self.rejected}],
        }


def _val(x: Any) -> str:
    return getattr(x, "value", x)


def majority_human_choice(annotations: Iterable[AnnotationLike]) -> str | None:
    votes = Counter(_val(a.preference_choice) for a in annotations if _val(a.annotator_type) == "HUMAN")
    if not votes:
        return None
    (top, n), *rest = votes.most_common()
    if rest and rest[0][1] == n:
        return None  # split decision
    return top


def pairs_for_sample(
    sample: SampleLike, annotations: Sequence[AnnotationLike], include_judge_labels: bool = False
) -> list[PreferencePair]:
    sid = str(sample.id)
    cand = {"CANDIDATE_A": sample.candidate_a, "CANDIDATE_B": sample.candidate_b}
    humans = [a for a in annotations if _val(a.annotator_type) == "HUMAN"]

    def pair(chosen: str, rejected: str, source: str) -> PreferencePair:
        return PreferencePair(sid, sample.system_prompt, sample.user_prompt, chosen, rejected, source)

    rewrites = [a for a in humans if (a.rewritten_chosen_response or "").strip()]
    if rewrites:
        golden = rewrites[-1].rewritten_chosen_response.strip()
        return [pair(golden, text, "human_rewrite") for text in cand.values() if text.strip() != golden]

    if humans:
        choice = majority_human_choice(humans)
        if choice in cand:
            other = "CANDIDATE_B" if choice == "CANDIDATE_A" else "CANDIDATE_A"
            return [pair(cand[choice], cand[other], "human")]
        return []

    if include_judge_labels:
        judged = [a for a in annotations if _val(a.annotator_type) == "LLM_JUDGE" and a.position_consistent]
        if judged and _val(judged[-1].preference_choice) in cand:
            choice = _val(judged[-1].preference_choice)
            other = "CANDIDATE_B" if choice == "CANDIDATE_A" else "CANDIDATE_A"
            return [pair(cand[choice], cand[other], "llm_judge")]
    return []


def build_preference_pairs(
    samples: Iterable[tuple[SampleLike, Sequence[AnnotationLike]]], include_judge_labels: bool = False
) -> list[PreferencePair]:
    out: list[PreferencePair] = []
    for sample, anns in samples:
        out.extend(pairs_for_sample(sample, anns, include_judge_labels))
    return out


# --------------------------------------------------------------------------- training

REQUIRED_ML_MODULES = ("torch", "transformers", "trl", "peft", "datasets")


class TrainingDependencyError(RuntimeError):
    pass


def ml_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in REQUIRED_ML_MODULES)


def cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


@dataclass
class DPOTrainingConfig:
    base_model: str
    output_dir: str
    beta: float = 0.1
    learning_rate: float = 5e-6
    num_train_epochs: float = 1.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_qlora: bool = False
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def run_dpo_training(
    pairs: Sequence[PreferencePair],
    cfg: DPOTrainingConfig,
    on_log: Callable[[dict], None] | None = None,
) -> dict:
    """Fine-tune `cfg.base_model` with LoRA (or 4-bit QLoRA on CUDA) using TRL's DPOTrainer."""
    if not ml_available():
        raise TrainingDependencyError(
            "Training requires the ML extra: pip install -e 'backend[ml]' (add '[qlora]' for 4-bit on Linux/CUDA)."
        )
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DPOConfig, DPOTrainer

    if len(pairs) < 2:
        raise ValueError("DPO needs at least 2 preference pairs")

    use_cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if cfg.use_qlora:
        if not use_cuda or importlib.util.find_spec("bitsandbytes") is None:
            raise TrainingDependencyError("QLoRA needs a CUDA GPU and bitsandbytes (pip install -e 'backend[qlora]').")
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)

    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    args = DPOConfig(
        output_dir=cfg.output_dir,
        beta=cfg.beta,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_length=cfg.max_length,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=cfg.seed,
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=use_cuda,
    )

    class _LogCallback(TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **_kw):
            if on_log and logs:
                on_log({"step": state.global_step, **{k: float(v) for k, v in logs.items()
                                                       if isinstance(v, (int, float))}})

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list([p.to_trl() for p in pairs]),
        processing_class=tokenizer,
        peft_config=lora,
        callbacks=[_LogCallback()],
    )
    result = trainer.train()
    adapter_dir = Path(cfg.output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return {
        "adapter_path": str(adapter_dir),
        "train_runtime_s": round(float(result.metrics.get("train_runtime", 0.0)), 2),
        "train_loss": float(result.training_loss),
        "log_history": trainer.state.log_history,
    }
