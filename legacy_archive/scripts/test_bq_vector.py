#!/usr/bin/env python3
"""Test script for BigQuery vector search functionality.

This script tests the BigQuery helper functions with sample queries.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils_bigquery import (
    vector_search,
    keyword_search,
    count_documents,
    get_table_info,
    get_vector_index_info,
    fetch_chunks_by_ids,
)
from utils import create_embedding_function


def test_vector_search():
    """Test vector search with a sample query."""
    print("\n" + "="*60)
    print("Test 1: Vector Search")
    print("="*60)

    # Sample query
    query_text = "What are the roofing requirements for asphalt shingles?"

    # Create embedding function
    try:
        embedding_fn = create_embedding_function()
        query_embedding = embedding_fn([query_text])[0]

        print(f"Query: {query_text}")
        print(f"Embedding dimension: {len(query_embedding)}")

        # Perform vector search
        results = vector_search(
            query_embedding=query_embedding,
            limit=5
        )

        print(f"\nResults: {len(results)} chunks found")
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Similarity: {result.get('similarity', 0):.4f}")
            print(f"Source: {result.get('source_url', 'N/A')}")
            print(f"Content: {result['content'][:200]}...")

        print("\n✓ Vector search test passed")
        return True

    except Exception as e:
        print(f"\n✗ Vector search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_keyword_search():
    """Test keyword search with sample terms."""
    print("\n" + "="*60)
    print("Test 2: Keyword Search")
    print("="*60)

    # Sample search terms
    terms = ["1507.9.6", "TABLE 1507", "asphalt shingle"]

    print(f"Search terms: {terms}")

    try:
        results = keyword_search(terms=terms, limit=5)

        print(f"\nResults: {len(results)} chunks found")
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Source: {result.get('source_url', 'N/A')}")
            print(f"Content: {result['content'][:200]}...")

        print("\n✓ Keyword search test passed")
        return True

    except Exception as e:
        print(f"\n✗ Keyword search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_count_and_info():
    """Test document count and table info retrieval."""
    print("\n" + "="*60)
    print("Test 3: Document Count & Table Info")
    print("="*60)

    try:
        # Count documents
        count = count_documents()
        print(f"Total documents: {count:,}")

        # Get table info
        info = get_table_info()
        print(f"\nTable Information:")
        print(f"  Project: {info.get('project')}")
        print(f"  Dataset: {info.get('dataset')}")
        print(f"  Table: {info.get('table')}")
        print(f"  Rows: {info.get('num_rows', 0):,}")
        print(f"  Size: {info.get('num_bytes', 0):,} bytes")
        print(f"  Schema fields: {info.get('schema_fields', 0)}")

        # Get index info
        indexes = get_vector_index_info()
        print(f"\nVector Indexes: {len(indexes)}")
        for idx in indexes:
            print(f"  - {idx.get('index_name')}: {idx.get('status')}")
            if idx.get('coverage_percentage'):
                print(f"    Coverage: {idx.get('coverage_percentage')}%")

        print("\n✓ Count and info test passed")
        return True

    except Exception as e:
        print(f"\n✗ Count and info test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fetch_by_ids():
    """Test fetching chunks by specific IDs."""
    print("\n" + "="*60)
    print("Test 4: Fetch Chunks by IDs")
    print("="*60)

    try:
        # First, get some IDs from a keyword search
        sample_results = keyword_search(terms=["roof"], limit=3)

        if not sample_results:
            print("⚠ No results to test with, skipping ID fetch test")
            return True

        chunk_ids = [r['chunk_id'] for r in sample_results]
        print(f"Fetching {len(chunk_ids)} chunks by ID...")

        results = fetch_chunks_by_ids(chunk_ids)

        print(f"\nResults: {len(results)} chunks retrieved")
        for result in results:
            print(f"  - {result['chunk_id']}: {result['content'][:100]}...")

        print("\n✓ Fetch by IDs test passed")
        return True

    except Exception as e:
        print(f"\n✗ Fetch by IDs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_filtered_search():
    """Test vector search with filters."""
    print("\n" + "="*60)
    print("Test 5: Filtered Vector Search")
    print("="*60)

    try:
        query_text = "roofing requirements"
        embedding_fn = create_embedding_function()
        query_embedding = embedding_fn([query_text])[0]

        # Test with source filter
        filters = {"source_contains": "ibc"}
        results = vector_search(
            query_embedding=query_embedding,
            limit=3,
            filters=filters
        )

        print(f"Query: {query_text}")
        print(f"Filters: {filters}")
        print(f"Results: {len(results)} chunks found")

        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {result.get('source_url', 'N/A')}")
            print(f"Content: {result['content'][:150]}...")

        print("\n✓ Filtered search test passed")
        return True

    except Exception as e:
        print(f"\n✗ Filtered search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("BigQuery Vector Store Tests")
    print("="*60)

    # Check environment
    project = os.getenv("BQ_PROJECT", "cal-rag-agent")
    dataset = os.getenv("BQ_DATASET", "calrag")
    table = os.getenv("BQ_TABLE", "documents")

    print(f"\nConfiguration:")
    print(f"  Project: {project}")
    print(f"  Dataset: {dataset}")
    print(f"  Table: {table}")

    # Run tests
    tests = [
        ("Count & Info", test_count_and_info),
        ("Vector Search", test_vector_search),
        ("Keyword Search", test_keyword_search),
        ("Fetch by IDs", test_fetch_by_ids),
        ("Filtered Search", test_filtered_search),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{test_name}' raised exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
