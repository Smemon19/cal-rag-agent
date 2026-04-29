from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import policy_badger_pro as p


def _chunk(doc: str, text: str) -> dict:
    tokens = p._words(text)
    return {
        "chunk_id": f"{doc}-{len(tokens)}",
        "doc": doc,
        "text": text,
        "text_lc": text.lower(),
        "token_counts": Counter(tokens),
        "char_count": len(text),
        "word_count": len(tokens),
    }


def test_chunk_text_by_paragraphs_bounds() -> None:
    para = " ".join(["policy"] * 60)
    text = "\n\n".join([para, para, para, para, para])
    chunks = p._chunk_text_by_paragraphs(text, min_words=120, max_words=180)
    assert chunks
    word_counts = [len(chunk.split()) for chunk in chunks]
    assert all(wc <= 180 for wc in word_counts)
    assert all(wc >= 120 for wc in word_counts[:-1])


def test_extract_keywords_removes_stop_words() -> None:
    keywords = p._extract_keywords("What is Raymond's overtime policy for weekend work?")
    assert "what" not in keywords
    assert "is" not in keywords
    assert "overtime" in keywords
    assert "policy" in keywords


def test_select_relevant_chunks_ranks_expected_doc() -> None:
    context = {
        "chunks": [
            _chunk("Overtime Policy.pdf", "Overtime policy overtime hours beyond forty are paid at 1.5x."),
            _chunk("Travel Policy.pdf", "Travel expenses require approval before booking flights."),
        ],
        "stats": {"documents_loaded": 2, "chunks_indexed": 2},
    }
    result = p._select_relevant_chunks(
        question="What is the overtime policy?",
        context=context,
        top_k=2,
        max_context_chars=2000,
    )
    assert result["chunks"]
    assert result["chunks"][0]["doc"] == "Overtime Policy.pdf"


def test_select_relevant_chunks_enforces_topk_and_char_cap() -> None:
    long_text = "overtime " * 120
    context = {
        "chunks": [
            _chunk("A.pdf", long_text),
            _chunk("B.pdf", long_text),
            _chunk("C.pdf", long_text),
        ],
        "stats": {"documents_loaded": 3, "chunks_indexed": 3},
    }
    result = p._select_relevant_chunks(
        question="overtime policy",
        context=context,
        top_k=2,
        max_context_chars=len(long_text) + 5,
    )
    assert len(result["chunks"]) == 1
    assert result["context_chars"] <= len(long_text) + 5


def test_ask_policy_pro_returns_deduped_sources(monkeypatch) -> None:
    class _FakeResponse:
        text = "Overtime is paid at 1.5x after 40 hours."

    class _FakeModel:
        def __init__(self, _name: str):
            pass

        def generate_content(self, _prompt: str):
            return _FakeResponse()

    monkeypatch.setattr(p.google.auth, "default", lambda: (object(), "proj"))
    monkeypatch.setattr(p, "vertexai_init", lambda **_kwargs: None)
    monkeypatch.setattr(p, "GenerativeModel", _FakeModel)

    context = {
        "chunks": [
            _chunk("Overtime Policy.pdf", "Overtime policy overtime applies after forty hours."),
            _chunk("Overtime Policy.pdf", "Overtime rate is one and one-half times regular pay."),
        ],
        "stats": {"documents_loaded": 1, "chunks_indexed": 2},
    }
    result = p.ask_policy_pro("What is the overtime policy?", context)
    assert "1.5x" in result["answer_text"]
    assert result["sources"] == ["Overtime Policy.pdf"]
    assert result["debug"]["context_chunks_used"] >= 1
