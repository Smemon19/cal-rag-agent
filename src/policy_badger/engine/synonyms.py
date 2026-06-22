"""Lightweight company vocabulary expansion before search planning."""

SYNONYM_MAP: dict[str, str] = {
    "pd": "professional development",
    "client dinner": "meal expense client meeting expense category",
    "holiday party": "voluntary social event",
    "proposal time": "bid proposal activity",
    "tqc": "total quality control",
    "teqc": "total quality control",
    "what do i put this under": "expense category",
    "what category": "expense category",
    "timesheet category": "activity type timesheet",
    "charge this": "expense category or billing category",
}


def expand_question(question: str) -> str:
    """Append known company terms while preserving the user's original wording."""
    original = question or ""
    normalized = original.lower()
    expansions = [mapped for key, mapped in SYNONYM_MAP.items() if key in normalized]
    if not expansions:
        return original
    return f"{original} {' '.join(expansions)}"
