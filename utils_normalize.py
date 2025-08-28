"""Normalization and expansion helpers for numeric/unit/section-aware search.

This module focuses on improving retrieval recall for queries that contain
numbers, units, inequality symbols, and direct section references.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Set


def normalize_inequalities(text: str) -> str:
    """Normalize common inequality symbols and spacing.

    Replacements:
    - ≥ -> >=
    - ≤ -> <=
    - Fix spaced operators like "> =" -> ">=", "< =" -> "<="
    """
    if not text:
        return text
    s = text
    s = s.replace("≥", ">=")
    s = s.replace("≤", "<=")
    # Normalize spaced operators
    s = re.sub(r">\s*=\s*", ">=", s)
    s = re.sub(r"<\s*=\s*", "<=", s)
    return s


_UNIT_SYNONYMS = {
    "mph": ["miles per hour"],
    "miles per hour": ["mph"],
    "lb": ["pounds"],
    "pounds": ["lb"],
    "kn": ["kilonewtons"],
    "kilonewtons": ["kN"],  # preserve case variant too
    "kN": ["kilonewtons"],
}


def _with_thousands(n: int) -> str:
    return f"{n:,}"


def _float_fmt(x: float) -> str:
    # Trim trailing zeros but keep one decimal if present in examples like 26.7
    s = f"{x:.1f}"
    return s


def _detect_numbers(text: str) -> List[str]:
    # Capture integers and decimals with optional thousands separators
    # Examples: 140, 6,000, 26.7
    nums = re.findall(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)", text)
    return nums


def _parse_float(num_str: str) -> float:
    try:
        return float(num_str.replace(",", ""))
    except Exception:
        return float("nan")


def expand_numeric_tokens(text: str) -> List[str]:
    """Generate numeric/unit variants from the input text.

    Examples for "140":
      - "140 mph", "140 miles per hour"
      - ">= 140 mph", "140 or greater"
      - "140 lb", "140 pounds", "140 kN", "140 kilonewtons" (literal, not converted)

    If a number is paired with a recognized unit in the text, also include a
    converted variant for lb <-> kN (force) using 1 lb ≈ 0.0044482216 kN.
    E.g., "6000 lb" -> "26.7 kN".
    """
    if not text:
        return []

    s = text
    numbers = _detect_numbers(s)
    if not numbers:
        return []

    variants: Set[str] = set()

    # Identify explicit number+unit pairs in the original text for conversions
    pair_pattern = re.compile(
        r"(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(?P<unit>mph|miles per hour|lb|pounds|kN|kn|kilonewtons)\b",
        flags=re.IGNORECASE,
    )
    pairs = [(m.group("num"), m.group("unit")) for m in pair_pattern.finditer(s)]

    # Build plain expansions for all numbers
    for num_raw in numbers:
        # Canonical integer form when possible
        num_val = _parse_float(num_raw)
        num_int = int(num_val) if num_val == int(num_val) else None

        # As-is and with thousands separator (if integer)
        if num_int is not None:
            variants.add(str(num_int))
            variants.add(_with_thousands(num_int))
        else:
            variants.add(num_raw)

        # mph variants
        mph_terms = ["mph", "miles per hour"]
        for u in mph_terms:
            variants.add(f"{num_raw} {u}")
        # Inequality phrasing
        variants.add(f">= {num_raw} mph")
        variants.add(f"{num_raw} or greater")

        # lb/kN literal attachments (non-converted)
        for u in ["lb", "pounds", "kN", "kilonewtons"]:
            variants.add(f"{num_raw} {u}")

    # Conversions for explicit pairs present in text
    for num_str, unit in pairs:
        unit_l = unit.lower()
        val = _parse_float(num_str)
        if val != val:  # NaN
            continue
        if unit_l in {"lb", "pounds"}:
            kn = val * 0.0044482216152605
            kn_s = _float_fmt(kn)
            variants.add(f"{kn_s} kN")
            variants.add(f"{kn_s} kilonewtons")
        elif unit_l in {"kn", "kn", "kilonewtons"}:  # normalize case
            # kN -> lb: 1 kN ≈ 224.80894387 lb
            lb = val * 224.80894387096
            # Round to nearest whole number for readability
            lb_int = int(round(lb))
            variants.add(f"{lb_int} lb")
            variants.add(f"{_with_thousands(lb_int)} pounds")

    return [v for v in variants if v]


def pair_unit_synonyms(tokens: Iterable[str]) -> List[str]:
    """Ensure unit synonyms are paired in the provided tokens.

    For any token that contains a unit keyword, add tokens with the synonym unit.
    E.g., if a token contains "mph", also include a version replacing it with
    "miles per hour".
    """
    out: Set[str] = set(tokens)
    for t in list(out):
        tl = t.lower()
        for unit, syns in _UNIT_SYNONYMS.items():
            if unit in tl:
                for syn in syns:
                    out.add(re.sub(unit, syn, t, flags=re.IGNORECASE))
    return list(out)


