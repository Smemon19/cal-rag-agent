"""Lightweight answer verifier for numeric/code token inclusion and citation.

Purely string-based, deterministic checks to ensure the drafted answer includes
the salient tokens observed in the retrieved context.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple

from utils_normalize import normalize_inequalities, pair_unit_synonyms, expand_numeric_tokens


SECTION_RE = re.compile(r"\b\d{3,4}(?:\.\d+)+\b")


def extract_key_tokens(context_text: str) -> Dict[str, List[str]]:
    """Extract sections, numeric-with-units, and standards from context text.

    - Sections: tokens like 1507.7, 1607.9, 1607.10.2, 1604.3
    - Numerics: examples include '6000 lb', '6,000 pounds', '26.7 kN', 'L/180', 'L/120', '4:12', '140 mph'
    - Standards: ASTM tags like 'ASTM D226 Type II', 'ASTM D4869', 'ASTM C406'
    """
    text = context_text or ""

    # Sections
    sections = SECTION_RE.findall(text)

    # Numbers with units and ratios
    nums: List[str] = []
    # L/### ratios
    nums += re.findall(r"\b[Ll]/\d+\b", text)
    # X:Y ratios (e.g., 4:12)
    nums += re.findall(r"\b\d+\s*:\s*\d+\b", text)
    # magnitude + unit (lb, pounds, kN, kilonewtons, mph, miles per hour)
    nums += re.findall(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*(?:lb|pounds|kN|kilonewtons|mph|miles per hour)\b", text, flags=re.IGNORECASE)
    nums += re.findall(r"\b\d+(?:\.\d+)?\s*(?:lb|pounds|kN|kilonewtons|mph|miles per hour)\b", text, flags=re.IGNORECASE)

    # Standards (ASTM ... [Type ...]) keep phrases compact
    standards: List[str] = []
    standards += re.findall(r"\bASTM\s+[A-Z]\d{3,4}(?:\s+Type\s+[A-Za-z0-9IVX]+)?\b", text)
    standards += re.findall(r"\bASTM\s+[A-Z]{1,2}\d{3,4}\b", text)

    # Deduplicate while preserving order
    def dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    return {
        "sections": dedupe(sections),
        "nums": dedupe(nums),
        "standards": dedupe(standards),
    }


def normalize_tokens(tokens: List[str]) -> List[str]:
    """Normalize inequality symbols and pair unit synonyms; preserve order, dedupe.

    Also augment numeric tokens with expanded variants (e.g., thousands separators,
    mph/miles per hour, lb/pounds, kN/kilonewtons) to make presence checks robust.
    """
    expanded: List[str] = []
    for t in tokens or []:
        t2 = normalize_inequalities(t)
        expanded.append(t2)
        # numeric expansions
        expanded.extend(expand_numeric_tokens(t2))
    # pair unit synonyms across all tokens
    expanded = pair_unit_synonyms(expanded)
    # Deduplicate preserving order (case-insensitive for stability)
    seen_ci = set()
    out: List[str] = []
    for s in expanded:
        key = s.lower()
        if key not in seen_ci:
            seen_ci.add(key)
            out.append(s)
    return out


def pick_salient_numbers(nums: List[str]) -> List[str]:
    """Select up to 5 salient numeric tokens likely to be limits/loads/ratios."""
    if not nums:
        return []
    patterns = [
        re.compile(r"\b[Ll]/\d+\b"),                # L/###
        re.compile(r"\b\d+\s*:\s*\d+\b"),         # X:Y ratios
        re.compile(r"\b\d{3,5}\s*(lb|pounds|kN|kilonewtons)\b", re.IGNORECASE),
        re.compile(r"\b\d{2,3}\s*(mph|miles per hour)\b", re.IGNORECASE),
    ]
    salient: List[str] = []
    for n in nums:
        for p in patterns:
            if p.search(n):
                salient.append(n)
                break
        if len(salient) >= 5:
            break
    # Fallback: if none matched, take first up to 3
    if not salient:
        return nums[:3]
    return salient[:5]


def _contains_any(answer_text: str, variants: List[str]) -> bool:
    a = (answer_text or "").lower()
    for v in variants:
        if v and v.lower() in a:
            return True
    return False


def verify_answer(answer_text: str, context_text: str) -> Dict[str, Any]:
    """Verify that the answer contains key tokens evident in the context.

    - All section tokens found in the context must appear in the answer.
    - For numerics: require presence of up to 3–5 salient numeric tokens.
    - For standards: if present in context, require presence.
    """
    extracted = extract_key_tokens(context_text)
    ctx_sections = extracted.get("sections", [])
    ctx_nums = extracted.get("nums", [])
    ctx_stds = extracted.get("standards", [])

    # Limit to salient numerics to avoid overly strict checks
    salient_nums = pick_salient_numbers(ctx_nums)

    missing_sections: List[str] = []
    missing_nums: List[str] = []
    missing_stds: List[str] = []

    # Sections must appear exactly (case-insensitive)
    for s in ctx_sections:
        if s and s.lower() not in (answer_text or "").lower():
            missing_sections.append(s)

    # Numbers: accept variants
    for n in salient_nums:
        variants = normalize_tokens([n])
        if not _contains_any(answer_text, variants):
            missing_nums.append(n)

    # Standards must appear literally (case-insensitive) to encourage quoting
    for st in ctx_stds:
        if st and st.lower() not in (answer_text or "").lower():
            missing_stds.append(st)

    ok = not (missing_sections or missing_nums or missing_stds)
    present = {
        "sections": [s for s in ctx_sections if s not in missing_sections],
        "nums": [n for n in salient_nums if n not in missing_nums],
        "standards": [st for st in ctx_stds if st not in missing_stds],
    }
    return {
        "ok": ok,
        "missing": {"sections": missing_sections, "nums": missing_nums, "standards": missing_stds},
        "present": present,
    }


