"""
Tell new policies apart from ones already published.

Uploading a document re-extracts every policy it contains, including the many
that are already in policies_v2 from a previous version of the same handbook.
Publishing that batch wholesale would duplicate them. This module compares each
staged candidate against the live table and labels it:

  new        — nothing comparable is published
  duplicate  — an existing policy already says substantially the same thing
  update     — same topic/subtopic, but the wording changed enough to look like
               a revision of that policy rather than a restatement

The comparison is deliberately lexical (token-set overlap) rather than a model
call: it runs over every existing row for every candidate, needs to be fast and
free, and must give the same answer twice so a reviewer can trust the labels.
The admin reviews the labels and decides what to publish; nothing is ingested
automatically.

Known limit: a revision that rewrites a policy in almost entirely new wording
scores as "new" rather than "update", because the only thresholds low enough to
catch it would also mislabel unrelated policies that merely share a topic. That
is the safe direction to fail — the admin ingests it and supersedes the old row
from the Policy Library, whereas a wrong "update" label invites replacing the
wrong policy. Duplicate detection, which is what stops a re-uploaded handbook
doubling the table, is not affected.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

from policy_badger.engine.db import run_query

# Overlap at or above this is treated as the same policy restated.
DUPLICATE_THRESHOLD = 0.82

# Below DUPLICATE_THRESHOLD but at or above this, with a matching topic, reads
# as a revision of the existing policy.
UPDATE_THRESHOLD = 0.45

# Text fields that carry the substance of a policy.
COMPARISON_FIELDS = ("condition_text", "action_text", "recommendation_text", "source_quote")

# Very common words carry no signal and would inflate every score.
STOPWORDS = frozenset("""
a an and are as at be been but by for from had has have if in into is it its
may must not of on or should so than that the their then there these they this
to was were will with within without you your employee employees policy
""".split())

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: Any) -> set[str]:
    """Normalised, de-stopworded token set for one piece of text."""
    words = _WORD_RE.findall(str(text or "").lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def _candidate_tokens(record: dict) -> set[str]:
    out: set[str] = set()
    for field in COMPARISON_FIELDS:
        out |= _tokens(record.get(field))
    return out


def similarity(left: set[str], right: set[str]) -> float:
    """
    Overlap coefficient, not Jaccard.

    A handbook section often restates a policy more verbosely than the stored
    row. Jaccard punishes that length difference and would call a genuine
    duplicate "new"; the overlap coefficient asks how much of the smaller text
    is contained in the larger, which is the question that matters here.
    """
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def load_existing_policies() -> list[dict]:
    """Active published policies, with their comparison tokens precomputed."""
    rows = run_query(
        """
        SELECT policy_id, topic, subtopic, condition_text, action_text, source_quote
        FROM policies_v2
        WHERE COALESCE(status, 'active') = 'active'
        """
    )
    for row in rows:
        row["_tokens"] = _candidate_tokens(row)
        row["_topic"] = str(row.get("topic") or "").strip().lower()
        row["_subtopic"] = str(row.get("subtopic") or "").strip().lower()
    return rows


def classify_candidate(candidate: dict, existing: Iterable[dict]) -> dict[str, Any]:
    """
    Compare one extracted candidate against the published set.

    Returns the verdict plus the best match, so the reviewer can see exactly
    which existing policy drove the decision.
    """
    cand_tokens = _candidate_tokens(candidate)
    cand_topic = str(candidate.get("topic") or "").strip().lower()
    cand_subtopic = str(candidate.get("subtopic") or "").strip().lower()

    best: Optional[dict] = None
    best_score = 0.0

    for row in existing:
        score = similarity(cand_tokens, row["_tokens"])
        # A shared topic is corroborating evidence, not proof on its own.
        if cand_topic and cand_topic == row["_topic"]:
            score += 0.08
            if cand_subtopic and cand_subtopic == row["_subtopic"]:
                score += 0.05
        if score > best_score:
            best_score = score
            best = row

    score = min(best_score, 1.0)

    if best is None or not cand_tokens:
        verdict = "new"
    elif score >= DUPLICATE_THRESHOLD:
        verdict = "duplicate"
    elif score >= UPDATE_THRESHOLD and cand_topic and cand_topic == best["_topic"]:
        verdict = "update"
    else:
        verdict = "new"

    return {
        "verdict": verdict,
        "score": round(score, 3),
        "matched_policy_id": str(best["policy_id"]) if best and verdict != "new" else None,
        "matched_topic": best.get("topic") if best and verdict != "new" else None,
        "matched_action_text": best.get("action_text") if best and verdict != "new" else None,
        "matched_condition_text": best.get("condition_text") if best and verdict != "new" else None,
    }


def review_batch(staged_rows: list[dict]) -> dict[str, Any]:
    """
    Label every staged extraction in a batch against what is already published.

    The returned items are what the admin review table renders. Only rows the
    admin then selects are published.
    """
    existing = load_existing_policies()

    items: list[dict] = []
    counts = {"new": 0, "duplicate": 0, "update": 0, "unpublishable": 0}

    for row in staged_rows:
        candidate = _as_dict(row.get("candidate_json"))

        # The extractor marks prose that carries no rule at all; there is
        # nothing to compare or publish.
        if not candidate.get("publishable", True):
            counts["unpublishable"] += 1
            items.append({
                "extraction_id": str(row.get("extraction_id")),
                "verdict": "unpublishable",
                "score": 0.0,
                "reason": "The extractor found no policy rule in this section.",
                "topic": candidate.get("topic"),
                "subtopic": candidate.get("subtopic"),
                "condition_text": candidate.get("condition_text"),
                "action_text": candidate.get("action_text"),
                "source_quote": candidate.get("source_quote"),
                "confidence": row.get("confidence"),
                "matched_policy_id": None,
            })
            continue

        verdict = classify_candidate(candidate, existing)
        counts[verdict["verdict"]] += 1
        items.append({
            "extraction_id": str(row.get("extraction_id")),
            "topic": candidate.get("topic"),
            "subtopic": candidate.get("subtopic"),
            "condition_text": candidate.get("condition_text"),
            "action_text": candidate.get("action_text"),
            "source_quote": candidate.get("source_quote"),
            "confidence": row.get("confidence"),
            **verdict,
        })

    # New policies first, then likely revisions, then the noise.
    order = {"new": 0, "update": 1, "duplicate": 2, "unpublishable": 3}
    items.sort(key=lambda i: (order.get(i["verdict"], 9), -(i.get("score") or 0)))

    return {"items": items, "counts": counts, "total": len(items)}
