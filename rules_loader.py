"""Loader and simple lookup helpers for rule snippets.

Rules are stored as small JSON files under rules/ with the schema described
in the README/task. We load all rules at module import and offer helpers to
locate them by section or by keyword tokens.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple


class RuleIndex:
    def __init__(self) -> None:
        self.by_section: Dict[str, List[Dict[str, Any]]] = {}
        self.search_keys: List[Tuple[str, Dict[str, Any]]] = []

    def add(self, rule: Dict[str, Any]) -> None:
        sec = str(rule.get("sec") or "").strip()
        if not sec:
            return
        self.by_section.setdefault(sec, []).append(rule)
        # Build search keys from title and item labels/keys
        title = str(rule.get("title") or "").strip().lower()
        if title:
            self.search_keys.append((title, rule))
        for item in rule.get("items", []) or []:
            for field in ("label", "key"):
                val = str(item.get(field) or "").strip().lower()
                if val:
                    self.search_keys.append((val, rule))


_RULES_INDEX = RuleIndex()


def load_all_rules(base_dir: str = "rules") -> Dict[str, List[Dict[str, Any]]]:
    """Load all JSON files from base_dir and return an index by section.

    Also populates an in-memory search index for keyword lookups.
    """
    _RULES_INDEX.by_section.clear()
    _RULES_INDEX.search_keys.clear()

    base = Path(base_dir)
    if not base.exists() or not base.is_dir():
        return _RULES_INDEX.by_section

    for path in base.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                rule = json.load(f)
            if isinstance(rule, dict) and rule.get("sec"):
                _RULES_INDEX.add(rule)
        except Exception as e:
            print(f"[rules] Failed to load {path}: {e}")

    return _RULES_INDEX.by_section


def find_rules_by_section(section_token: str) -> List[Dict[str, Any]]:
    """Return rules for a section token, or empty list when not found."""
    return list(_RULES_INDEX.by_section.get(section_token, []))


def find_rules_by_keywords(tokens: List[str]) -> List[Tuple[Dict[str, Any], int]]:
    """Return rules ranked by the number of token hits in title/items.

    Very simple scoring: for each rule, count how many tokens are substrings of
    any of its searchable fields. Returns a list of (rule, score) sorted by
    descending score and then by title.
    """
    if not tokens:
        return []
    tok_l = [t.lower() for t in tokens if t]
    scores: Dict[int, int] = {}
    rules: List[Dict[str, Any]] = []
    # Map rules to an index to accumulate scores
    rule_to_idx: Dict[int, int] = {}

    for text, rule in _RULES_INDEX.search_keys:
        rid = id(rule)
        if rid not in rule_to_idx:
            rule_to_idx[rid] = len(rules)
            rules.append(rule)
            scores[rule_to_idx[rid]] = 0
        for t in tok_l:
            if t and t in text:
                scores[rule_to_idx[rid]] += 1

    ranked: List[Tuple[Dict[str, Any], int]] = []
    for idx, rule in enumerate(rules):
        score = scores.get(idx, 0)
        if score > 0:
            ranked.append((rule, score))

    ranked.sort(key=lambda x: (-x[1], str(x[0].get("title"))))
    return ranked


# Eager load on import
load_all_rules("rules")


