#!/usr/bin/env python3
"""Verify BigQuery table setup for cal-rag-agent migration.

This script checks:
1. BigQuery table exists and has data
2. Schema is correct (includes embedding column and required metadata)
3. Vector index exists
4. Sample queries work
"""

import os
import sys
from google.cloud import bigquery
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_bigquery_setup():
    """Verify BigQuery table and index configuration."""

    # Configuration
    project_id = os.getenv("BQ_PROJECT", "cal-rag-agent")
    dataset_id = os.getenv("BQ_DATASET", "calrag")
    table_id = os.getenv("BQ_TABLE", "documents")
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"

    print(f"\n{'='*60}")
    print(f"BigQuery Setup Verification")
    print(f"{'='*60}")
    print(f"Project: {project_id}")
    print(f"Dataset: {dataset_id}")
    print(f"Table: {table_id}")
    print(f"Full table ID: {full_table_id}")
    print(f"{'='*60}\n")

    try:
        # Initialize BigQuery client
        client = bigquery.Client(project=project_id)
        print("✓ BigQuery client initialized successfully")

        # Check if table exists
        try:
            table = client.get_table(full_table_id)
            print(f"✓ Table exists: {full_table_id}")
        except Exception as e:
            print(f"✗ Table not found: {e}")
            return False

        # Get row count
        query = f"SELECT COUNT(*) as count FROM `{full_table_id}`"
        result = client.query(query).result()
        row_count = list(result)[0]["count"]
        print(f"✓ Row count: {row_count:,}")

        if row_count == 0:
            print("⚠ Warning: Table is empty")

        # Check schema
        print(f"\n{'='*60}")
        print("Schema:")
        print(f"{'='*60}")

        required_columns = ["chunk_id", "content", "embedding", "source_url", "metadata"]
        found_columns = {field.name: field.field_type for field in table.schema}

        for col in required_columns:
            if col in found_columns:
                print(f"✓ {col:20s} ({found_columns[col]})")
            else:
                print(f"✗ {col:20s} (MISSING)")

        # Check for embedding column specifically
        if "embedding" in found_columns:
            if found_columns["embedding"] in ["FLOAT64", "ARRAY", "REPEATED"]:
                print(f"✓ Embedding column type is compatible")
            else:
                print(f"⚠ Embedding column type may need verification: {found_columns['embedding']}")

        # List all columns
        print(f"\n{'='*60}")
        print("All columns:")
        print(f"{'='*60}")
        for field in table.schema:
            print(f"  {field.name:30s} {field.field_type:15s} {field.mode:10s}")

        # Check for vector index
        print(f"\n{'='*60}")
        print("Vector Indexes:")
        print(f"{'='*60}")

        index_query = f"""
        SELECT
            index_name,
            table_name,
            index_status,
            coverage_percentage
        FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.VECTOR_INDEXES`
        WHERE table_name = '{table_id}'
        """

        try:
            index_result = client.query(index_query).result()
            indexes = list(index_result)

            if indexes:
                for idx in indexes:
                    print(f"✓ Index: {idx['index_name']}")
                    print(f"  Status: {idx['index_status']}")
                    print(f"  Coverage: {idx.get('coverage_percentage', 'N/A')}%")
            else:
                print("⚠ No vector indexes found")
                print("  Note: You may need to create an index for optimal performance")
        except Exception as e:
            print(f"⚠ Could not query vector indexes: {e}")

        # Sample data
        print(f"\n{'='*60}")
        print("Sample Data (first 3 rows):")
        print(f"{'='*60}")

        sample_query = f"""
        SELECT
            chunk_id,
            LEFT(content, 100) as content_preview,
            source_url,
            ARRAY_LENGTH(embedding) as embedding_dim
        FROM `{full_table_id}`
        LIMIT 3
        """

        sample_result = client.query(sample_query).result()
        for i, row in enumerate(sample_result, 1):
            print(f"\nRow {i}:")
            print(f"  Chunk ID: {row['chunk_id']}")
            print(f"  Content: {row['content_preview']}...")
            print(f"  Source: {row['source_url']}")
            print(f"  Embedding dim: {row.get('embedding_dim', 'N/A')}")

        # Check metadata structure (if it's JSON)
        print(f"\n{'='*60}")
        print("Metadata Sample:")
        print(f"{'='*60}")

        metadata_query = f"""
        SELECT metadata
        FROM `{full_table_id}`
        WHERE metadata IS NOT NULL
        LIMIT 1
        """

        try:
            metadata_result = client.query(metadata_query).result()
            for row in metadata_result:
                import json
                if isinstance(row['metadata'], str):
                    metadata = json.loads(row['metadata'])
                else:
                    metadata = row['metadata']

                print("Metadata keys found:")
                for key in sorted(metadata.keys() if hasattr(metadata, 'keys') else []):
                    print(f"  - {key}")
                break
        except Exception as e:
            print(f"⚠ Could not parse metadata: {e}")

        print(f"\n{'='*60}")
        print("✓ Verification complete!")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\n✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_bigquery_setup()
    sys.exit(0 if success else 1)
