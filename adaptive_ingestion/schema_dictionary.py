"""Canonical schema dictionary with alias normalization and overrides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from policy_engine.db import run_query


@dataclass(frozen=True)
class ConceptDef:
    name: str
    type_hint: str
    synonyms: tuple[str, ...]
    reserved_names: tuple[str, ...]


class SchemaDictionary:
    """Loads canonical dictionary and DB-backed overrides."""

    def __init__(self, dictionary_path: Path | None = None):
        base = Path(__file__).resolve().parent
        self.dictionary_path = dictionary_path or (base / "schema_dictionary" / "policy_schema_dictionary.yaml")
        payload = self._load_yaml()
        self._mapping_rules: dict[str, str] = {
            str(k).strip().lower(): str(v).strip()
            for k, v in (payload.get("mapping_rules") or {}).items()
        }
        self._concepts = self._build_concepts(payload)
        self._alias_to_canonical = self._build_alias_map(self._concepts, self._mapping_rules)

    def _load_yaml(self) -> dict[str, Any]:
        if not self.dictionary_path.exists():
            raise FileNotFoundError(f"Dictionary not found: {self.dictionary_path}")
        with self.dictionary_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        if not isinstance(payload, dict):
            raise ValueError("Dictionary YAML must load as an object.")
        return payload

    @staticmethod
    def _build_concepts(payload: dict[str, Any]) -> dict[str, ConceptDef]:
        concepts = {}
        for name, raw in (payload.get("concepts") or {}).items():
            raw = raw or {}
            concepts[name] = ConceptDef(
                name=name,
                type_hint=str(raw.get("type_hint") or "text"),
                synonyms=tuple(str(x).strip().lower() for x in (raw.get("synonyms") or [])),
                reserved_names=tuple(str(x).strip().lower() for x in (raw.get("reserved_names") or [])),
            )
        return concepts

    @staticmethod
    def _build_alias_map(concepts: dict[str, ConceptDef], mapping_rules: dict[str, str]) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for name, concept in concepts.items():
            aliases[name.lower()] = name
            for syn in concept.synonyms:
                if syn:
                    aliases[syn] = name
        for alias, target in mapping_rules.items():
            aliases[alias] = target
        return aliases

    def refresh_overrides(self) -> None:
        """Apply DB-backed override aliases and reserved names."""
        try:
            rows = run_query(
                """
                SELECT canonical_name, aliases, reserved_names, mapping_rules
                FROM schema_dictionary_overrides
                WHERE active = TRUE
                """,
            )
        except Exception:
            return

        for row in rows:
            canonical = str(row.get("canonical_name") or "").strip()
            if not canonical:
                continue
            aliases = row.get("aliases") or []
            rules = row.get("mapping_rules") or {}
            if canonical not in self._concepts:
                self._concepts[canonical] = ConceptDef(canonical, "text", tuple(), tuple())
            for alias in aliases:
                alias_s = str(alias).strip().lower()
                if alias_s:
                    self._alias_to_canonical[alias_s] = canonical
            if isinstance(rules, dict):
                for k, v in rules.items():
                    self._alias_to_canonical[str(k).strip().lower()] = str(v).strip()

    def normalize_name(self, raw_name: str) -> str:
        """Map an observed field name to a canonical concept when possible."""
        key = (raw_name or "").strip().lower()
        if not key:
            return ""
        return self._alias_to_canonical.get(key, key)

    def canonical_fields(self) -> set[str]:
        return set(self._concepts.keys())

    def type_hint_for(self, name: str) -> str:
        canonical = self.normalize_name(name)
        concept = self._concepts.get(canonical)
        return concept.type_hint if concept else "text"

    def is_reserved(self, field_name: str) -> bool:
        key = (field_name or "").strip().lower()
        if not key:
            return False
        for concept in self._concepts.values():
            if key in concept.reserved_names:
                return True
        return False

