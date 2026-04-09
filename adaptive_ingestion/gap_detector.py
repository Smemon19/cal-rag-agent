"""Deterministic schema gap detection."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from adaptive_ingestion.schema_dictionary import SchemaDictionary


@dataclass
class Gap:
    concept: str
    gap_type: str
    reason: str
    frequency: int
    examples: list[str]


class GapDetector:
    def __init__(self, dictionary: SchemaDictionary):
        self.dictionary = dictionary

    def detect(self, staged_rows: list[dict], existing_columns: set[str]) -> list[Gap]:
        counts = Counter()
        examples: dict[str, list[str]] = defaultdict(list)
        type_hints: dict[str, str] = {}

        for row in staged_rows:
            unmapped = row.get("unmapped_concepts_json") or []
            if isinstance(unmapped, str):
                try:
                    unmapped = json.loads(unmapped)
                except Exception:
                    unmapped = []
            for item in unmapped:
                raw = str(item.get("normalized_label") or item.get("raw_label") or "").strip()
                if not raw:
                    continue
                canonical = self.dictionary.normalize_name(raw)
                counts[canonical] += int(item.get("frequency_in_batch") or 1)
                if len(examples[canonical]) < 3:
                    examples[canonical].extend([str(v) for v in (item.get("observed_values") or [])[:2]])
                type_hints[canonical] = self.dictionary.type_hint_for(canonical)

        gaps: list[Gap] = []
        for concept, freq in counts.items():
            if concept in existing_columns:
                gaps.append(Gap(concept, "new_value", "existing field received new values", freq, examples[concept]))
                continue
            hint = type_hints.get(concept, "text")
            if hint == "relationship":
                gap_type = "new_relationship"
                reason = "multi-valued/relationship concept should be modeled in child table"
            elif hint in {"numeric", "bool", "enum_like"}:
                gap_type = "new_scalar"
                reason = "queryable scalar concept missing from schema"
            else:
                gap_type = "text_only"
                reason = "low-structure concept should remain text unless repeated"
            gaps.append(Gap(concept, gap_type, reason, freq, examples[concept]))
        gaps.sort(key=lambda g: g.frequency, reverse=True)
        return gaps

