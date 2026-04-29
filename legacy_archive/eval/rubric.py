from __future__ import annotations

import json
import re
from typing import Dict, Any, List, Tuple


SECTION_RE = re.compile(r"\b\d{3,4}(?:\.\d+)+\b")


def score_item(answer: str, item: Dict[str, Any]) -> Dict[str, Any]:
    text = (answer or "").lower()
    must = [m.lower() for m in item.get("must_contain", [])]
    nice = [n.lower() for n in item.get("nice_to_have", [])]

    missing: List[str] = []
    hits_must = 0
    for m in must:
        if m and m.lower() in text:
            hits_must += 1
        else:
            missing.append(m)

    # accuracy 0–2
    if hits_must == len(must) and len(must) > 0:
        accuracy = 2
    elif hits_must == 0 and len(must) > 0:
        accuracy = 0
    else:
        accuracy = 1

    # completeness 0–2 based on nice_to_have count
    hits_nice = sum(1 for n in nice if n and n.lower() in text)
    if hits_nice >= 2:
        completeness = 2
    elif hits_nice == 1:
        completeness = 1
    else:
        completeness = 0

    # grounding 0–2
    grounding = 0
    if SECTION_RE.search(answer or ""):
        grounding = 2
    elif re.search(r"\b(IBC|Table)\b", answer or "", flags=re.IGNORECASE):
        grounding = 1

    total = accuracy + completeness + grounding
    return {
        "qid": item.get("qid"),
        "accuracy": accuracy,
        "completeness": completeness,
        "grounding": grounding,
        "total": total,
        "missing": missing,
    }


def aggregate(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not scores:
        return {"avg_total": 0, "pass_rate": 0.0}
    totals = [s["total"] for s in scores]
    avg_total = sum(totals) / len(totals)
    pass_rate = sum(1 for t in totals if t >= 5) / len(totals)  # threshold: 5/6
    return {"avg_total": avg_total, "pass_rate": pass_rate}


