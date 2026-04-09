"""Batch-level schema proposal consolidation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from uuid import uuid4

from adaptive_ingestion.gap_detector import Gap
from policy_engine.db import execute


def classify_migration(gap_type: str, concept: str) -> str:
    """Classify migration risk policy."""
    if gap_type == "new_value":
        return "safe_auto"
    if gap_type in {"new_scalar", "new_relationship"}:
        return "approval_required"
    if concept in {"status", "policy_id", "section_id"}:
        return "disallowed"
    return "approval_required"


@dataclass
class SchemaProposal:
    request_id: str
    batch_id: str
    migration_class: str
    items: list[dict]


class SchemaPlanner:
    def build_proposals(self, *, batch_id: str, gaps: list[Gap]) -> list[SchemaProposal]:
        grouped: dict[str, list[dict]] = {"safe_auto": [], "approval_required": [], "disallowed": []}
        for gap in gaps:
            migration_class = classify_migration(gap.gap_type, gap.concept)
            grouped[migration_class].append(asdict(gap))
        proposals: list[SchemaProposal] = []
        for klass, items in grouped.items():
            if not items:
                continue
            proposals.append(
                SchemaProposal(
                    request_id=f"scr_{uuid4().hex}",
                    batch_id=batch_id,
                    migration_class=klass,
                    items=items,
                )
            )
        return proposals

    def persist_proposals(self, proposals: list[SchemaProposal]) -> None:
        for proposal in proposals:
            execute(
                """
                INSERT INTO schema_change_requests (request_id, batch_id, proposal_json, migration_class, decision_status)
                VALUES (%s, %s, %s::jsonb, %s, 'pending')
                ON CONFLICT (request_id) DO NOTHING
                """,
                (
                    proposal.request_id,
                    proposal.batch_id,
                    json.dumps({"items": proposal.items}),
                    proposal.migration_class,
                ),
            )

