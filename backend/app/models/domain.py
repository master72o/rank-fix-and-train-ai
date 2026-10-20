"""ORM models: datasets, prompt samples, annotations and training runs."""

import enum
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

EMBEDDING_DIM = 384

# JSONB + pgvector on PostgreSQL, plain JSON on SQLite.
JSONType = JSON().with_variant(JSONB(), "postgresql")
EmbeddingType = JSON().with_variant(Vector(EMBEDDING_DIM), "postgresql")


class TaskType(str, enum.Enum):
    DPO_PREFERENCE = "DPO_PREFERENCE"
    SFT_EDIT = "SFT_EDIT"
    KTO_BINARY = "KTO_BINARY"


class SampleStatus(str, enum.Enum):
    PENDING = "PENDING"
    ANNOTATED = "ANNOTATED"
    DISPUTED = "DISPUTED"
    EXPORTED = "EXPORTED"


class AnnotatorType(str, enum.Enum):
    HUMAN = "HUMAN"
    LLM_JUDGE = "LLM_JUDGE"


class PreferenceChoice(str, enum.Enum):
    CANDIDATE_A = "CANDIDATE_A"
    CANDIDATE_B = "CANDIDATE_B"
    TIE = "TIE"
    BOTH_BAD = "BOTH_BAD"


class RunStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    task_type: Mapped[TaskType] = mapped_column(Enum(TaskType, name="task_type"), nullable=False)
    rubric_definition: Mapped[dict] = mapped_column(JSONType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    samples: Mapped[list["PromptSample"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class PromptSample(Base):
    __tablename__ = "prompt_samples"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    category: Mapped[str | None] = mapped_column(String(50))
    system_prompt: Mapped[str | None] = mapped_column(Text)
    user_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingType)

    candidate_a: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_a_model: Mapped[str | None] = mapped_column(String(100))
    candidate_a_logprob_entropy: Mapped[float | None] = mapped_column(Float)
    candidate_b: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_b_model: Mapped[str | None] = mapped_column(String(100))
    candidate_b_logprob_entropy: Mapped[float | None] = mapped_column(Float)

    semantic_divergence_score: Mapped[float | None] = mapped_column(Float)
    active_learning_priority: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    status: Mapped[SampleStatus] = mapped_column(
        Enum(SampleStatus, name="sample_status"), default=SampleStatus.PENDING, index=True
    )
    # Hidden reference metadata (e.g. gold preference for the bundled golden set). Never shown to annotators.
    meta: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    dataset: Mapped[Dataset] = relationship(back_populates="samples")
    annotations: Mapped[list["Annotation"]] = relationship(
        back_populates="sample", cascade="all, delete-orphan", order_by="Annotation.created_at"
    )


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    sample_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("prompt_samples.id", ondelete="CASCADE"), index=True)
    annotator_id: Mapped[str] = mapped_column(String(100), nullable=False)
    annotator_type: Mapped[AnnotatorType] = mapped_column(Enum(AnnotatorType, name="annotator_type"))
    preference_choice: Mapped[PreferenceChoice] = mapped_column(Enum(PreferenceChoice, name="preference_choice"))
    rewritten_chosen_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    rubric_scores: Mapped[dict] = mapped_column(JSONType, nullable=False)
    error_flags: Mapped[list] = mapped_column(JSONType, default=list)
    critique_rationale: Mapped[str | None] = mapped_column(Text)
    # LLM judge only: did the verdict survive the A/B -> B/A position swap?
    position_consistent: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    time_spent_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    sample: Mapped[PromptSample] = relationship(back_populates="annotations")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus, name="run_status"), default=RunStatus.QUEUED)
    base_model: Mapped[str] = mapped_column(String(200), nullable=False)
    config: Mapped[dict] = mapped_column(JSONType, default=dict)
    num_pairs: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    eval_report: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    adapter_path: Mapped[str | None] = mapped_column(String(500))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
