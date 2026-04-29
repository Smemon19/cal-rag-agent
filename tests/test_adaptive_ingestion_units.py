from __future__ import annotations

import sys
from pathlib import Path
import types

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_ingestion.contracts import (
    CandidateJson,
    FieldMappingValue,
    StagingPayload,
    UnmappedConcept,
)
from adaptive_ingestion.gap_detector import GapDetector
from adaptive_ingestion.policy_extractor import PolicyExtractor
from adaptive_ingestion.schema_dictionary import SchemaDictionary
from adaptive_ingestion.schema_planner import SchemaPlanner, classify_migration
from adaptive_ingestion.section_chunker import SectionChunker
from adaptive_ingestion.pipeline import AdaptiveIngestionPipeline, IngestionDocumentInput


def test_section_chunker_splits_markdown() -> None:
    chunker = SectionChunker()
    parts = chunker.split_sections("# A\nrule 1\n## B\nrule 2")
    assert len(parts) == 2
    assert parts[0][0] == "A"
    assert "rule 1" in parts[0][1]


def test_staging_contract_accepts_valid_payload() -> None:
    payload = StagingPayload(
        candidate_json=CandidateJson(
            candidate_id="cand_1",
            document_id="doc_1",
            section_id="sec_1",
            chunk_type="policy_rule",
            publishable=True,
            summary="Sample policy summary",
            topic="policy",
            subtopic="general",
            condition_text="if condition",
            action_text="must do action",
            recommendation_text=None,
            entities=["Raymond"],
            source_quote="quote",
            reason_if_not_publishable=None,
            rule_text="rule",
            confidence=0.8,
            extractor_version="v1",
        ),
        mapped_fields_json={
            "topic": FieldMappingValue(value="policy", confidence=0.7, provenance="unit-test"),
        },
        unmapped_concepts_json=[
            UnmappedConcept(
                raw_label="annual_limit",
                normalized_label="amount_threshold",
                observed_values=["$5000"],
                suggested_classification="new_scalar",
                frequency_in_batch=2,
            )
        ],
    )
    assert payload.candidate_json.extractor_version == "v1"
    assert "topic" in payload.mapped_fields_json


def test_extractor_maps_annual_cap_alias(monkeypatch) -> None:
    monkeypatch.setenv("USE_LLM_SEMANTIC_EXTRACTION", "0")
    dictionary = SchemaDictionary()
    extractor = PolicyExtractor(dictionary)
    out = extractor.extract(
        document_id="doc_1",
        section_id="sec_1",
        section_text="The annual cap is $5,000 per year and approval is required.",
    )
    mapped = out.payload.mapped_fields_json
    assert "amount_threshold" in mapped
    assert "approval_required" in mapped
    assert out.payload.candidate_json.chunk_type in {"policy_rule", "procedure", "discussion"}
    assert out.payload.candidate_json.confidence >= 0.55


def test_extractor_marks_filler_non_publishable(monkeypatch) -> None:
    monkeypatch.setenv("USE_LLM_SEMANTIC_EXTRACTION", "0")
    dictionary = SchemaDictionary()
    extractor = PolicyExtractor(dictionary)
    out = extractor.extract(
        document_id="doc_1",
        section_id="sec_1",
        section_text="Thanks, bye.",
    )
    assert out.payload.candidate_json.chunk_type == "filler"
    assert out.payload.candidate_json.publishable is False


def test_gap_detector_classifies_new_scalar() -> None:
    dictionary = SchemaDictionary()
    detector = GapDetector(dictionary)
    staged = [
        {
            "unmapped_concepts_json": [
                {
                    "raw_label": "annual_limit",
                    "normalized_label": "amount_threshold",
                    "observed_values": ["5000"],
                    "frequency_in_batch": 3,
                }
            ]
        }
    ]
    gaps = detector.detect(staged, existing_columns={"policy_category", "status"})
    assert gaps
    assert gaps[0].concept == "amount_threshold"
    assert gaps[0].gap_type in {"new_scalar", "new_value"}


def test_schema_planner_groups_migration_classes() -> None:
    assert classify_migration("new_value", "department") == "safe_auto"
    assert classify_migration("new_scalar", "annual_cap") == "approval_required"

    planner = SchemaPlanner()
    proposals = planner.build_proposals(
        batch_id="bat_1",
        gaps=[],
    )
    assert proposals == []


def test_pipeline_extracts_pdf_content_with_ocr(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "policy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    fake_pdf_loader = types.ModuleType("pdf_loader.pdf_loader")

    def fake_process_pdf(_pdf_path, _output_json, _image_dir, chunk_size=500):  # noqa: ARG001
        return [{"text": "Scanned policy text from image OCR."}]

    fake_pdf_loader.process_pdf = fake_process_pdf
    monkeypatch.setitem(sys.modules, "pdf_loader.pdf_loader", fake_pdf_loader)

    pipe = AdaptiveIngestionPipeline()
    text = pipe._extract_content_for_input(
        IngestionDocumentInput(source_uri=f"file://{pdf_path}", content="")
    )
    assert "OCR" in text

