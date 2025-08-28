"""Utilities for detecting and parsing Markdown tables and extracting values.

This module adds a lightweight Markdown table parser suitable for tables
commonly found in code- or documentation-derived chunks. It is intentionally
conservative and avoids bringing in heavy Markdown parsing frameworks.
"""

from __future__ import annotations

from typing import List, Optional
import re
import pandas as pd


_TABLE_HEADER_RE = re.compile(r"^\|.*\|\s*$")
_TABLE_DELIM_RE = re.compile(r"^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|\s*$")


def _split_row(row: str) -> List[str]:
    """Split a Markdown table row into cells.

    - Strips leading/trailing pipes
    - Preserves empty cells
    - Trims surrounding whitespace per cell
    """
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    # Split on literal pipe; Markdown tables do not escape pipes typically in our data
    parts = [p.strip() for p in row.split("|")]
    return parts


def _coerce_header_names(cells: List[str]) -> List[str]:
    """Generate clean header names, filling blanks with positional names."""
    headers: List[str] = []
    for i, c in enumerate(cells):
        name = (c or "").strip()
        if not name:
            name = f"col_{i+1}"
        headers.append(name)
    # Deduplicate by appending suffixes if repeated
    seen: dict[str, int] = {}
    deduped: List[str] = []
    for h in headers:
        cnt = seen.get(h, 0)
        if cnt == 0:
            deduped.append(h)
        else:
            deduped.append(f"{h}_{cnt+1}")
        seen[h] = cnt + 1
    return deduped


def parse_markdown_table(md: str) -> List[pd.DataFrame]:
    """Detect Markdown tables in text and parse each into a pandas DataFrame.

    The parser looks for blocks in the form:
        | H1 | H2 |
        | --- | --- |
        | v1 | v2 |

    Returns a list of DataFrames (one per detected table). Tables with
    malformed headers or delimiter lines are skipped.
    """
    if not md:
        return []

    lines = md.splitlines()
    dataframes: List[pd.DataFrame] = []

    i = 0
    n = len(lines)
    while i < n:
        # Find a potential header row
        if _TABLE_HEADER_RE.match(lines[i] or ""):
            # Next line must be a delimiter row
            if i + 1 < n and _TABLE_DELIM_RE.match(lines[i + 1] or ""):
                header_cells = _split_row(lines[i])
                headers = _coerce_header_names(header_cells)

                # Collect data rows until a non-table line encountered
                j = i + 2
                rows: List[List[str]] = []
                while j < n and _TABLE_HEADER_RE.match(lines[j] or ""):
                    rows.append(_split_row(lines[j]))
                    j += 1

                # Normalize row lengths to headers
                normalized_rows: List[List[Optional[str]]] = []
                for r in rows:
                    if len(r) < len(headers):
                        r = r + [None] * (len(headers) - len(r))
                    elif len(r) > len(headers):
                        r = r[: len(headers)]
                    normalized_rows.append(r)

                try:
                    df = pd.DataFrame(normalized_rows, columns=headers)
                    # Keep only non-empty tables (at least one non-empty value)
                    if not df.empty and df.notna().any().any():
                        dataframes.append(df)
                except Exception:
                    # Skip malformed tables gracefully
                    pass

                i = j
                continue
        i += 1

    return dataframes


def _material_match(cell_value: Optional[str], material: str) -> bool:
    if cell_value is None:
        return False
    a = (cell_value or "").strip().lower()
    b = (material or "").strip().lower()
    if not a or not b:
        return False
    # Substring match is adequate for material names (e.g., "Slate" vs "Slate shingles")
    return b in a or a in b


def pick_underlayment(df: pd.DataFrame, material: str, wind_mph: int) -> Optional[str]:
    """Return the underlayment requirement cell for a given material and wind speed.

    Heuristics:
    - Row identification: first column that contains the material string (case-insensitive substring)
    - Column selection: prefer a column name indicating the wind band. If none exist,
      fallback to the right-most non-empty cell in the row.

    Wind band:
      - < 140 mph  -> pick a column whose header mentions '< 140', 'less than 140', 'up to 139', or similar
      - >= 140 mph -> pick a column whose header mentions '>= 140', '140 or greater', '140+', etc.
    """
    if df is None or df.empty:
        return None

    # Identify row index by scanning columns for a material match
    row_idx: Optional[int] = None
    for ri in range(len(df)):
        for col in df.columns:
            try:
                val = df.iloc[ri][col]
            except Exception:
                continue
            if _material_match(None if pd.isna(val) else str(val), material):
                row_idx = ri
                break
        if row_idx is not None:
            break

    if row_idx is None:
        return None

    # Determine target wind band
    want_high = wind_mph >= 140

    def header_score(h: str) -> int:
        hs = (h or "").strip().lower()
        score = 0
        if want_high:
            if ">=" in hs or "≥" in hs:
                score += 3
            if re.search(r"\b140\b", hs):
                score += 2
            if any(k in hs for k in ["greater", "or greater", "140+", "high wind", ">= 140"]):
                score += 2
        else:
            if "<" in hs or "≤" in hs:
                score += 3
            if re.search(r"\b(13[0-9]|1[01][0-9]|[0-9]{2})\b", hs):
                # mentions sub-140 numbers
                score += 1
            if any(k in hs for k in ["less than", "up to", "< 140", "<= 139"]):
                score += 2
        # Penalize clearly non-wind columns like material/type notes
        if any(k in hs for k in ["material", "roof", "covering", "description"]):
            score -= 2
        return score

    # Score all headers and pick the best non-empty cell
    best_col: Optional[str] = None
    best_score: int = -10**9
    for col in df.columns:
        val = df.iloc[row_idx][col]
        if pd.isna(val) or str(val).strip() == "":
            continue
        score = header_score(str(col))
        # Prefer later columns with same score (common structure: winds escalate to the right)
        if score > best_score or (score == best_score and best_col is not None and list(df.columns).index(col) > list(df.columns).index(best_col)):
            best_col = col
            best_score = score

    # Fallback: right-most non-empty cell
    if best_col is None:
        for col in reversed(list(df.columns)):
            val = df.iloc[row_idx][col]
            if not (pd.isna(val) or str(val).strip() == ""):
                best_col = col
                break

    if best_col is None:
        return None

    cell = df.iloc[row_idx][best_col]
    return None if pd.isna(cell) else str(cell).strip()


