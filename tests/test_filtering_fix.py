
import pytest
from rag_agent import _build_structured_extracts

def test_filtering_removed():
    """
    Verify that _build_structured_extracts does NOT filter out chunks 
    just because they lack the exact keywords from the question.
    """
    # Question asks about "height"
    question = "What is the max height?"
    
    # Document talks about "vertical limit" (synonym, but no keyword match for "height")
    doc_content = "The vertical limit for this structure is 50 feet. See Section 1507.2."
    metadata = {
        "section_path": "Chapter 15 > Section 1507.2",
        "title": "Roof Assemblies",
        "source_url": "http://example.com/ibc"
    }
    
    merged_results = {
        "documents": [[doc_content]],
        "metadatas": [[metadata]]
    }
    
    # Run the function
    context_text, entries = _build_structured_extracts(question, merged_results)
    
    # Assertions
    # 1. The document should be present in the output
    assert "vertical limit" in context_text.lower(), "The document content should be in the context"
    assert "1507.2" in context_text, "The section number should be in the context"
    
    # 2. Entries should not be empty
    assert len(entries) > 0, "Should have extracted an entry"
    assert entries[0]['section'] == "1507.2"

def test_filtering_still_keeps_matches():
    """
    Verify that it still keeps things that DO match (sanity check).
    """
    question = "What is the max height?"
    doc_content = "The max height is 50 feet. Section 1507.2."
    metadata = {"section_path": "1507.2"}
    
    merged_results = {
        "documents": [[doc_content]],
        "metadatas": [[metadata]]
    }
    
    context_text, entries = _build_structured_extracts(question, merged_results)
    assert "max height" in context_text.lower()
    assert len(entries) > 0

if __name__ == "__main__":
    # Manual run if pytest not available
    try:
        test_filtering_removed()
        print("test_filtering_removed PASSED")
        test_filtering_still_keeps_matches()
        print("test_filtering_still_keeps_matches PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
