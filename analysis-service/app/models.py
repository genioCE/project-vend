from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

EntityType = Literal["person", "place", "organization", "concept", "spiritual"]


class TypedEntity(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    type: EntityType


def _coerce_entity(v: Union[str, dict, "TypedEntity"]) -> "TypedEntity":
    """Accept plain strings (v4 data) or dicts/TypedEntity (v6 data)."""
    if isinstance(v, TypedEntity):
        return v
    if isinstance(v, str):
        return TypedEntity(name=v[:256], type="concept")
    if isinstance(v, dict):
        if "name" in v and isinstance(v["name"], str):
            v = {**v, "name": v["name"][:256]}
        return TypedEntity(**v)
    raise ValueError(f"Cannot coerce {type(v)} to TypedEntity")

DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


class SourceSpan(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    source_file: str | None = Field(default=None, max_length=512)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    excerpt: str | None = Field(default=None, max_length=280)

    @model_validator(mode="after")
    def validate_range(self) -> "SourceSpan":
        if self.end_char < self.start_char:
            raise ValueError("end_char must be greater than or equal to start_char")
        return self


class Provenance(BaseModel):
    chunk_ids: list[str] = Field(default_factory=list, max_length=200)
    spans: list[SourceSpan] = Field(default_factory=list, max_length=200)

    @field_validator("chunk_ids")
    @classmethod
    def normalize_chunk_ids(cls, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = value.strip()
            if not item:
                continue
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped


class AnalysisVersion(BaseModel):
    schema_version: str
    prompt_version: str
    model_version: str
    mock: bool = True


class SummarizeEntryRequest(BaseModel):
    entry_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=100000)
    entry_date: str | None = Field(default=None, pattern=DATE_PATTERN)
    source_file: str | None = Field(default=None, max_length=512)
    chunk_ids: list[str] = Field(default_factory=list, max_length=200)
    max_summary_sentences: int = Field(default=3, ge=1, le=8)

    @field_validator("entry_id", "text", "source_file")
    @classmethod
    def trim_required_fields(cls, value: str | None) -> str | None:
        if value is None:
            return value
        trimmed = " ".join(value.strip().split())
        if not trimmed:
            raise ValueError("value must not be blank")
        return trimmed


class SummarizeEntryResponse(BaseModel):
    analysis_id: str
    entry_id: str
    granularity: Literal["entry"]
    summary: str
    highlights: list[str]
    key_terms: list[str]
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    provenance: Provenance
    version: AnalysisVersion


class StateLabelRequest(BaseModel):
    entry_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=100000)
    entry_date: str | None = Field(default=None, pattern=DATE_PATTERN)
    source_file: str | None = Field(default=None, max_length=512)
    chunk_ids: list[str] = Field(default_factory=list, max_length=200)

    @field_validator("entry_id", "text", "source_file")
    @classmethod
    def trim_fields(cls, value: str | None) -> str | None:
        if value is None:
            return value
        trimmed = " ".join(value.strip().split())
        if not trimmed:
            raise ValueError("value must not be blank")
        return trimmed


StateDimension = Literal[
    "valence",
    "activation",
    "agency",
    "certainty",
    "relational_openness",
    "self_trust",
    "time_orientation",
    "integration",
]


class StateScoreRange(BaseModel):
    min: float = Field(default=-1.0, le=0.0)
    max: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "StateScoreRange":
        if self.min >= self.max:
            raise ValueError("score range min must be less than max")
        return self


class StateDimensionProfile(BaseModel):
    dimension: StateDimension
    score: float = Field(ge=-1.0, le=1.0)
    low_anchor: str
    high_anchor: str
    label: str
    evidence_spans: list[SourceSpan] = Field(default_factory=list, max_length=30)


class StateProfile(BaseModel):
    score_range: StateScoreRange = Field(default_factory=StateScoreRange)
    dimensions: list[StateDimensionProfile] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def validate_dimensions(self) -> "StateProfile":
        dimension_names = [item.dimension for item in self.dimensions]
        if len(set(dimension_names)) != len(dimension_names):
            raise ValueError("state profile dimensions must be unique")
        return self


class ObservedTextSignal(BaseModel):
    signal_id: str = Field(min_length=1, max_length=256)
    signal: str = Field(min_length=1, max_length=256)
    category: Literal[
        "lexical",
        "pattern",
        "modal",
        "temporal",
        "relational",
        "structural",
    ]
    direction: Literal["low", "high", "neutral"]
    dimensions: list[StateDimension] = Field(min_length=1, max_length=8)
    weight: float = Field(ge=0.0, le=1.0)
    evidence_spans: list[SourceSpan] = Field(default_factory=list, max_length=30)


class InferredStateLabel(BaseModel):
    dimension: StateDimension
    label: str
    score: float = Field(ge=-1.0, le=1.0)
    rationale: str
    supporting_signal_ids: list[str] = Field(default_factory=list, max_length=200)
    confidence: float = Field(ge=0.0, le=1.0)


class DimensionConfidence(BaseModel):
    dimension: StateDimension
    value: float = Field(ge=0.0, le=1.0)


class StateConfidence(BaseModel):
    overall: float = Field(ge=0.0, le=1.0)
    by_dimension: list[DimensionConfidence] = Field(min_length=8, max_length=8)


class StateLabelResponse(BaseModel):
    analysis_id: str
    entry_id: str
    state_profile: StateProfile
    observed_text_signals: list[ObservedTextSignal]
    inferred_state_labels: list[InferredStateLabel]
    confidence: StateConfidence
    provenance: Provenance
    version: AnalysisVersion


class EntryChunk(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=100000)
    source_file: str | None = Field(default=None, max_length=512)

    @field_validator("chunk_id", "text", "source_file")
    @classmethod
    def trim_chunk_fields(cls, value: str | None) -> str | None:
        if value is None:
            return value
        trimmed = " ".join(value.strip().split())
        if not trimmed:
            raise ValueError("value must not be blank")
        return trimmed


class EntrySummaryGenerateRequest(BaseModel):
    entry_id: str = Field(min_length=1, max_length=256)
    entry_date: str | None = Field(default=None, pattern=DATE_PATTERN)
    source_file: str | None = Field(default=None, max_length=512)
    chunks: list[EntryChunk] = Field(min_length=1, max_length=500)
    force_regenerate: bool = False
    provider: Literal["auto", "mock", "local", "hybrid", "ollama", "anthropic", "finetuned"] | None = None

    @field_validator("entry_id", "source_file")
    @classmethod
    def trim_generate_fields(cls, value: str | None) -> str | None:
        if value is None:
            return value
        trimmed = " ".join(value.strip().split())
        if not trimmed:
            raise ValueError("value must not be blank")
        return trimmed


class EntrySummaryProcessingMetadata(BaseModel):
    model_version: str
    prompt_version: str
    schema_version: str
    created_at: str
    provider: str
    mock: bool = True


class EntrySummaryRecord(BaseModel):
    entry_id: str
    entry_date: str | None = None
    source_file: str | None = None
    short_summary: str
    detailed_summary: str
    themes: list[str] = Field(default_factory=list, max_length=30)
    entities: list[TypedEntity] = Field(default_factory=list, max_length=50)
    decisions_actions: list[str] = Field(default_factory=list, max_length=50)
    state_profile: StateProfile
    provenance: Provenance
    processing: EntrySummaryProcessingMetadata

    @field_validator("entities", mode="before")
    @classmethod
    def coerce_entities(cls, v: list) -> list[TypedEntity]:
        return [_coerce_entity(item) for item in v]


class StateLabelProcessingMetadata(BaseModel):
    model_version: str
    prompt_version: str
    schema_version: str
    created_at: str
    provider: str
    mock: bool = True


class StateLabelRecord(BaseModel):
    entry_id: str
    entry_date: str | None = None
    source_file: str | None = None
    state_profile: StateProfile
    observed_text_signals: list[ObservedTextSignal]
    inferred_state_labels: list[InferredStateLabel]
    confidence: StateConfidence
    provenance: Provenance
    processing: StateLabelProcessingMetadata


class StateLabelGenerateRequest(BaseModel):
    entry_id: str = Field(min_length=1, max_length=256)
    entry_date: str | None = Field(default=None, pattern=DATE_PATTERN)
    source_file: str | None = Field(default=None, max_length=512)
    chunks: list[EntryChunk] = Field(min_length=1, max_length=500)
    force_regenerate: bool = False
    provider: Literal["auto", "mock", "local", "hybrid", "ollama", "anthropic", "finetuned"] | None = None

    @field_validator("entry_id", "source_file")
    @classmethod
    def trim_state_label_fields(cls, value: str | None) -> str | None:
        if value is None:
            return value
        trimmed = " ".join(value.strip().split())
        if not trimmed:
            raise ValueError("value must not be blank")
        return trimmed


class RetrievalHit(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    source_file: str | None = Field(default=None, max_length=512)
    excerpt: str = Field(min_length=1, max_length=2000)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("chunk_id", "excerpt", "source_file")
    @classmethod
    def trim_retrieval_fields(cls, value: str | None) -> str | None:
        if value is None:
            return value
        trimmed = " ".join(value.strip().split())
        if not trimmed:
            raise ValueError("value must not be blank")
        return trimmed


class GraphSignal(BaseModel):
    subject: str = Field(min_length=1, max_length=256)
    relation: str = Field(min_length=1, max_length=128)
    object: str = Field(min_length=1, max_length=256)
    weight: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("subject", "relation", "object")
    @classmethod
    def trim_graph_fields(cls, value: str) -> str:
        trimmed = " ".join(value.strip().split())
        if not trimmed:
            raise ValueError("value must not be blank")
        return trimmed


class ContextPacketRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    date_start: str | None = Field(default=None, pattern=DATE_PATTERN)
    date_end: str | None = Field(default=None, pattern=DATE_PATTERN)
    retrieval_hits: list[RetrievalHit] = Field(default_factory=list, max_length=50)
    graph_signals: list[GraphSignal] = Field(default_factory=list, max_length=50)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def trim_query(cls, value: str) -> str:
        trimmed = " ".join(value.strip().split())
        if not trimmed:
            raise ValueError("query must not be blank")
        return trimmed

    @model_validator(mode="after")
    def validate_dates(self) -> "ContextPacketRequest":
        if self.date_start and self.date_end and self.date_start > self.date_end:
            raise ValueError("date_start must be less than or equal to date_end")
        return self


class ContextRetrievalItem(BaseModel):
    chunk_id: str
    source_file: str | None = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    rationale: str


class ContextGraphItem(BaseModel):
    subject: str
    relation: str
    object: str
    weight: float = Field(ge=0.0, le=1.0)


class ContextPacketResponse(BaseModel):
    packet_id: str
    query: str
    temporal_focus: str
    retrieval_context: list[ContextRetrievalItem]
    graph_context: list[ContextGraphItem]
    context_brief: str
    provenance: Provenance
    version: AnalysisVersion
