from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_ingestion.schema_dictionary import SchemaDictionary


def test_dictionary_normalizes_synonyms() -> None:
    d = SchemaDictionary()
    assert d.normalize_name("annual_cap") == "amount_threshold"
    assert d.normalize_name("yearly cap") == "amount_threshold"
    assert "policy_category" in d.canonical_fields()

