"""
Per-user answer settings.

The settings page used to write to localStorage only, so preferences never
followed a user between devices and nothing on the server could read them.
These are stored per user and consumed by the formatter when answering.
"""

from __future__ import annotations

import json
from typing import Any

from policy_badger.engine.db import execute, run_query

# Mirrors the control set rendered by settings.html. Anything not listed here is
# dropped on save, so a stale or hand-crafted client cannot inject arbitrary
# keys into the prompt.
DEFAULT_SETTINGS: dict[str, Any] = {
    "prompts": [
        "What is the deadline for submitting timesheets?",
        "How do I submit an expense report?",
        "How do I request paid time off?",
        "Do I need approval for professional development?",
    ],
    "styles": ["Concise", "Cite sources", "Step-by-step"],
    "toggles": {"sources": True, "honest": True, "counts": True},
    "preset": "balanced",
    "format": "structured",
    "length": 1,
    "precision": 1,
    "reading": "plain",
}

ALL_STYLES = [
    "Concise", "Detailed", "Step-by-step", "Cite sources", "Plain language",
    "Formal", "Friendly", "Bullet points", "Lead with yes/no", "Flag exceptions",
    "Name the approver", "Include deadlines", "Show dollar limits", "Compare options",
]

VALID_PRESETS = {"balanced", "strict", "exploratory"}
VALID_FORMATS = {"structured", "prose", "bullets"}
VALID_READING = {"plain", "standard", "formal"}

MAX_PROMPTS = 8
MAX_PROMPT_LEN = 200


def _clamp_int(value: Any, low: int, high: int, fallback: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return fallback


def sanitize_settings(raw: Any) -> dict[str, Any]:
    """Coerce arbitrary client input into a known-good settings dict."""
    if not isinstance(raw, dict):
        return json.loads(json.dumps(DEFAULT_SETTINGS))

    out = json.loads(json.dumps(DEFAULT_SETTINGS))

    prompts = raw.get("prompts")
    if isinstance(prompts, list):
        cleaned = []
        for p in prompts[:MAX_PROMPTS]:
            if isinstance(p, str) and p.strip():
                cleaned.append(p.strip()[:MAX_PROMPT_LEN])
        out["prompts"] = cleaned

    styles = raw.get("styles")
    if isinstance(styles, list):
        out["styles"] = [s for s in ALL_STYLES if s in styles]

    toggles = raw.get("toggles")
    if isinstance(toggles, dict):
        out["toggles"] = {
            key: bool(toggles.get(key, default))
            for key, default in DEFAULT_SETTINGS["toggles"].items()
        }

    if raw.get("preset") in VALID_PRESETS:
        out["preset"] = raw["preset"]
    if raw.get("format") in VALID_FORMATS:
        out["format"] = raw["format"]
    if raw.get("reading") in VALID_READING:
        out["reading"] = raw["reading"]

    out["length"] = _clamp_int(raw.get("length"), 0, 2, DEFAULT_SETTINGS["length"])
    out["precision"] = _clamp_int(raw.get("precision"), 0, 10, DEFAULT_SETTINGS["precision"])

    return out


def get_user_settings(user_id: str) -> dict[str, Any]:
    """Return stored settings for a user, falling back to defaults."""
    rows = run_query("SELECT settings FROM user_settings WHERE user_id = %s", (user_id,))
    if not rows:
        return json.loads(json.dumps(DEFAULT_SETTINGS))

    stored = rows[0].get("settings")
    if isinstance(stored, str):
        try:
            stored = json.loads(stored)
        except json.JSONDecodeError:
            stored = None
    return sanitize_settings(stored)


def save_user_settings(user_id: str, raw: Any) -> dict[str, Any]:
    """Validate and persist settings, returning what was actually stored."""
    clean = sanitize_settings(raw)
    execute(
        """
        INSERT INTO user_settings (user_id, settings)
        VALUES (%s, %s)
        ON CONFLICT (user_id) DO UPDATE
            SET settings = EXCLUDED.settings, updated_at = now()
        """,
        (user_id, json.dumps(clean)),
    )
    return clean


def build_answer_directives(settings: dict[str, Any]) -> str:
    """
    Turn stored settings into prompt instructions for the formatter.

    Returns an empty string when settings match the defaults, so the existing
    prompt is left untouched for users who never opened the settings page.
    """
    if not settings:
        return ""

    lines: list[str] = []

    length = settings.get("length", 1)
    if length == 0:
        lines.append("- LENGTH: Keep the answer short. Two or three lines per section at most.")
    elif length == 2:
        lines.append("- LENGTH: Be thorough. Cover edge cases and related rules the rows support.")

    fmt = settings.get("format")
    if fmt == "prose":
        lines.append(
            "- FORMAT: Write flowing prose instead of the sectioned layout. "
            "Do not emit the section headers."
        )
    elif fmt == "bullets":
        lines.append("- FORMAT: Use terse bullet points throughout, including under Direct Answer.")

    reading = settings.get("reading")
    if reading == "plain":
        lines.append("- READING LEVEL: Use plain, everyday language. Expand jargon on first use.")
    elif reading == "formal":
        lines.append("- READING LEVEL: Use formal, policy-document register.")

    preset = settings.get("preset")
    if preset == "strict":
        lines.append(
            "- PRECISION: Only state what the rows explicitly support. "
            "If a detail is absent, say it is not specified rather than inferring it."
        )
    elif preset == "exploratory":
        lines.append(
            "- PRECISION: Surface related policies that may be relevant even when they do not "
            "answer the question directly, clearly labelled as related."
        )

    toggles = settings.get("toggles") or {}
    if toggles.get("sources"):
        lines.append("- Cite the originating policy topic or document for each substantive claim.")
    if toggles.get("honest"):
        lines.append("- State plainly when the retrieved rows do not fully answer the question.")
    if toggles.get("counts"):
        lines.append("- When several policies apply, say how many were found.")

    styles = settings.get("styles") or []
    style_rules = {
        "Concise": "be concise",
        "Detailed": "include supporting detail",
        "Step-by-step": "break actions into numbered steps",
        "Cite sources": "cite the source policy",
        "Plain language": "avoid jargon",
        "Formal": "keep a formal tone",
        "Friendly": "keep a warm, approachable tone",
        "Bullet points": "prefer bullets over paragraphs",
        "Lead with yes/no": "open with a direct yes or no when the question permits",
        "Flag exceptions": "call out exceptions prominently",
        "Name the approver": "name the approving role whenever the rows identify one",
        "Include deadlines": "state any deadlines explicitly",
        "Show dollar limits": "state any dollar limits explicitly",
        "Compare options": "contrast the available options when more than one applies",
    }
    active = [style_rules[s] for s in styles if s in style_rules]
    if active:
        lines.append("- STYLE: " + "; ".join(active) + ".")

    if not lines:
        return ""
    return "\n\nUser answer preferences (apply these without contradicting the evidence):\n" + "\n".join(lines)
