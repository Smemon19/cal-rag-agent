from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_ingestion.migration_generator import MigrationGenerator
from adaptive_ingestion.schema_planner import classify_migration


def test_classify_migration_policy() -> None:
    assert classify_migration("new_value", "department") == "safe_auto"
    assert classify_migration("new_scalar", "receipt_required") == "approval_required"
    assert classify_migration("new_relationship", "approver") == "approval_required"


def test_generate_sql_for_scalar_and_relationship() -> None:
    gen = MigrationGenerator()
    proposal = {
        "proposal_json": {
            "items": [
                {"concept": "receipt_required", "gap_type": "new_scalar"},
                {"concept": "approver", "gap_type": "new_relationship"},
            ]
        }
    }
    sqls = gen.generate(proposal)
    assert any("ALTER TABLE policies_v2" in s for s in sqls)
    assert any("CREATE TABLE IF NOT EXISTS policy_approver" in s for s in sqls)

