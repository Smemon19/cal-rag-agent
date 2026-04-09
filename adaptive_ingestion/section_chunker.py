"""Section/chunk extraction and persistence."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from uuid import uuid4

from policy_engine.db import run_query, transaction


@dataclass
class SectionRow:
    section_id: str
    document_id: str
    heading: str
    section_text: str
    ordinal: int


class SectionChunker:
    @staticmethod
    def _sections_columns() -> dict[str, str]:
        rows = run_query(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='sections'
            """
        )
        return {str(r["column_name"]): str(r["data_type"]) for r in rows}

    def split_sections(self, text: str) -> list[tuple[str, str]]:
        """Split by markdown-like headings; fallback to paragraph groups."""
        text = (text or "").strip()
        if not text:
            return []

        heading_re = re.compile(r"^(#{1,3}\s+.+)$", re.MULTILINE)
        matches = list(heading_re.finditer(text))
        if not matches:
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
            return [(f"Section {i+1}", p) for i, p in enumerate(paragraphs)]

        out: list[tuple[str, str]] = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            heading = m.group(1).lstrip("#").strip()
            body = text[start:end].strip()
            if body:
                out.append((heading, body))
        return out

    def persist_sections(self, *, document_id: str, text: str) -> list[SectionRow]:
        sections = self.split_sections(text)
        cols = self._sections_columns()
        rows: list[SectionRow] = []
        for idx, (heading, body) in enumerate(sections, start=1):
            section_id = f"sec_{uuid4().hex}"
            chunk_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

            doc_val: object = document_id
            if cols.get("document_id") in {"integer", "bigint", "smallint"}:
                doc_val = int(document_id)

            row_values: dict[str, object] = {
                "document_id": doc_val,
                "heading": heading,
                "section_text": body,
                "content": body,
                "ordinal": idx,
                "source_locator": f"section:{idx}",
                "chunk_hash": chunk_hash,
            }
            if cols.get("section_id") not in {"integer", "bigint", "smallint"}:
                row_values["section_id"] = section_id

            insert_cols = [c for c in row_values.keys() if c in cols]
            placeholders = ", ".join(["%s"] * len(insert_cols))
            values = tuple(row_values[c] for c in insert_cols)

            with transaction() as cur:
                cur.execute(
                    f"""
                    INSERT INTO sections ({", ".join(insert_cols)})
                    VALUES ({placeholders})
                    RETURNING section_id
                    """,
                    values,
                )
                returned = cur.fetchone() or {}
                returned_sid = returned.get("section_id")

            final_sid = str(returned_sid if returned_sid is not None else section_id)
            rows.append(
                SectionRow(
                    section_id=final_sid,
                    document_id=document_id,
                    heading=heading,
                    section_text=body,
                    ordinal=idx,
                )
            )
        return rows

