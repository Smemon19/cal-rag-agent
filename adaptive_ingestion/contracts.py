"""Strict contracts for staging payloads."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


ChunkType = Literal[
    "transcript_metadata",
    "filler",
    "discussion",
    "recommendation",
    "policy_rule",
    "policy_list",
    "procedure",
    "definition",
    "faq",
    "action_item",
]


class CandidateJson(BaseModel):
    candidate_id: str
    document_id: str
    section_id: str
    chunk_type: ChunkType
    publishable: bool
    summary: str
    topic: str | None = None
    subtopic: str | None = None
    condition_text: str | None = None
    action_text: str | None = None
    recommendation_text: str | None = None
    entities: list[str] = Field(default_factory=list)
    source_quote: str
    reason_if_not_publishable: str | None = None
    rule_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    extractor_version: str
    evidence_spans: list[dict] = Field(default_factory=list)
    normalization_notes: str | None = None


class FieldMappingValue(BaseModel):
    value: str | bool | int | float
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: str


class UnmappedConcept(BaseModel):
    raw_label: str
    normalized_label: str
    observed_values: list[str] = Field(default_factory=list)
    suggested_classification: Literal["new_value", "new_scalar", "new_relationship", "text_only"]
    frequency_in_batch: int = Field(ge=1)


class StagingPayload(BaseModel):
    candidate_json: CandidateJson
    mapped_fields_json: dict[str, FieldMappingValue]
    unmapped_concepts_json: list[UnmappedConcept]

    @field_validator("mapped_fields_json")
    @classmethod
    def validate_mapped_keys(cls, value: dict[str, FieldMappingValue]) -> dict[str, FieldMappingValue]:
        for key in value.keys():
            if not key.strip():
                raise ValueError("Mapped field keys must be non-empty.")
        return value

