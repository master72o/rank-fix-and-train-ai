"""Pydantic v2 request/response payloads."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.domain import AnnotatorType, PreferenceChoice, RunStatus, SampleStatus, TaskType


class ErrorFlag(str, enum.Enum):
    HALLUCINATION = "HALLUCINATION"
    FACTUAL_ERROR = "FACTUAL_ERROR"
    REASONING_ERROR = "REASONING_ERROR"
    CODE_BUG = "CODE_BUG"
    UNSAFE = "UNSAFE"
    OVER_REFUSAL = "OVER_REFUSAL"
    INSTRUCTION_VIOLATION = "INSTRUCTION_VIOLATION"
    VERBOSITY = "VERBOSITY"


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ----------------------------------------------------------------------------- datasets

class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    task_type: TaskType = TaskType.DPO_PREFERENCE
    rubric_definition: dict[str, str] | None = None


class DatasetOut(ORM):
    id: uuid.UUID
    name: str
    task_type: TaskType
    rubric_definition: dict
    created_at: datetime | None = None


# ----------------------------------------------------------------------------- prompts

class PromptIngestItem(BaseModel):
    user_prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    category: str | None = Field(default=None, max_length=50)
    candidate_a: str | None = None
    candidate_a_model: str | None = None
    candidate_a_logprob_entropy: float | None = Field(default=None, ge=0)
    candidate_b: str | None = None
    candidate_b_model: str | None = None
    candidate_b_logprob_entropy: float | None = Field(default=None, ge=0)
    meta: dict = Field(default_factory=dict)


class PromptIngestRequest(BaseModel):
    dataset_id: uuid.UUID
    items: list[PromptIngestItem] = Field(min_length=1, max_length=500)
    skip_duplicates: bool = True


class PromptIngestResponse(BaseModel):
    created: int
    sample_ids: list[uuid.UUID]
    skipped_duplicates: list[str]


class PromptSampleOut(ORM):
    """Annotator-facing view. Hidden reference metadata (`meta`) is deliberately excluded."""

    id: uuid.UUID
    dataset_id: uuid.UUID
    category: str | None
    system_prompt: str | None
    user_prompt: str
    candidate_a: str
    candidate_a_model: str | None
    candidate_a_logprob_entropy: float | None
    candidate_b: str
    candidate_b_model: str | None
    candidate_b_logprob_entropy: float | None
    semantic_divergence_score: float | None
    active_learning_priority: float
    status: SampleStatus


class QueueStats(BaseModel):
    dataset_id: uuid.UUID | None
    total: int
    by_status: dict[str, int]
    human_annotations: int
    judge_annotations: int
    review_threshold: float
    routing: dict[str, float | int]
    priority_histogram: list[dict[str, float | int]]


# ----------------------------------------------------------------------------- annotations

class AnnotationCreate(BaseModel):
    sample_id: uuid.UUID
    annotator_id: str = Field(min_length=1, max_length=100)
    preference_choice: PreferenceChoice
    rewritten_chosen_response: str | None = None
    rubric_scores: dict[str, float] = Field(default_factory=dict)
    error_flags: list[ErrorFlag] = Field(default_factory=list)
    critique_rationale: str | None = None
    time_spent_ms: int | None = Field(default=None, ge=0)

    @field_validator("rubric_scores")
    @classmethod
    def _scores_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        for k, score in v.items():
            if not 1 <= score <= 5:
                raise ValueError(f"rubric score {k!r} must be between 1 and 5")
        return v


class AnnotationOut(ORM):
    id: uuid.UUID
    sample_id: uuid.UUID
    annotator_id: str
    annotator_type: AnnotatorType
    preference_choice: PreferenceChoice
    rewritten_chosen_response: str | None
    rubric_scores: dict
    error_flags: list
    critique_rationale: str | None
    position_consistent: bool | None
    time_spent_ms: int | None
    created_at: datetime | None = None


class AnnotationCreated(BaseModel):
    annotation: AnnotationOut
    sample_status: SampleStatus


# ----------------------------------------------------------------------------- judge

class JudgeRunRequest(BaseModel):
    dataset_id: uuid.UUID
    sample_ids: list[uuid.UUID] | None = None
    limit: int = Field(default=50, ge=1, le=1000)
    only_human_annotated: bool = False


class JobAccepted(BaseModel):
    job: str
    queued: int
    mode: str  # celery | background
    run_id: uuid.UUID | None = None


class Disagreement(BaseModel):
    sample_id: uuid.UUID
    user_prompt: str
    human: str
    judge: str
    position_consistent: bool | None


class AgreementReport(BaseModel):
    dataset_id: uuid.UUID
    judge_name: str | None
    n_pairs: int
    cohen_kappa: float | None = None
    kappa_ci_low: float | None = None
    kappa_ci_high: float | None = None
    kappa_se: float | None = None
    kappa_interpretation: str | None = None
    percent_agreement: float | None = None
    labels: list[str]
    confusion_matrix: list[list[int]] | None = None
    position_consistency_rate: float | None
    judge_verdicts: int
    judge_label_distribution: dict[str, int]
    human_label_distribution: dict[str, int]
    judge_gold_accuracy: float | None
    gold_items: int
    disagreements: list[Disagreement]
    message: str | None = None


# ----------------------------------------------------------------------------- training

class TrainingRunCreate(BaseModel):
    dataset_id: uuid.UUID
    base_model: str | None = None
    beta: float = Field(default=0.1, gt=0, le=1)
    learning_rate: float = Field(default=5e-6, gt=0, le=1e-2)
    num_train_epochs: float = Field(default=1.0, gt=0, le=20)
    per_device_batch_size: int = Field(default=1, ge=1, le=64)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=256)
    max_length: int = Field(default=1024, ge=128, le=8192)
    lora_r: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    use_qlora: bool = False
    include_judge_labels: bool = False
    eval_holdout: int = Field(default=8, ge=0, le=200)


class TrainingRunOut(ORM):
    id: uuid.UUID
    dataset_id: uuid.UUID
    status: RunStatus
    base_model: str
    config: dict
    num_pairs: int
    metrics: dict
    eval_report: dict | None
    adapter_path: str | None
    error: str | None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class PairPreview(BaseModel):
    dataset_id: uuid.UUID
    num_pairs: int
    sources: dict[str, int]
    examples: list[dict]


class Capabilities(BaseModel):
    ml_available: bool
    cuda_available: bool
    base_model: str
    judge_model: str
    generator_backend: str
    embedding_backend: str
    celery_enabled: bool
