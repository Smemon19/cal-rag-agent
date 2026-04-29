#!/usr/bin/env python3
"""Initialize BigQuery schema for Cal RAG Agent.

This script creates:
1. The dataset (default: calrag)
2. The table (default: documents) with the correct schema
3. The vector index on the embedding column
"""

import os
import sys
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, Conflict

def init_bigquery():
    # Configuration
    project_id = os.getenv("BQ_PROJECT", "cal-rag-agent")
    dataset_id = os.getenv("BQ_DATASET", "calrag")
    table_id = os.getenv("BQ_TABLE", "documents")
    
    # Schema definition
    schema = [
        bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        bigquery.SchemaField("source_url", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("source_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("section_path", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("headers", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("page_number", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("title", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("mime_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("char_count", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("word_count", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("content_preview", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("embedding_backend", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("embedding_model", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("inserted_at", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
        # Keep created_at for backward compat if needed, or just use inserted_at
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
    ]

    print(f"Initializing BigQuery resources in project: {project_id}")
    
    client = bigquery.Client(project=project_id)

    # 1. Create Dataset
    dataset_ref = f"{project_id}.{dataset_id}"
    try:
        client.get_dataset(dataset_ref)
        print(f"✓ Dataset exists: {dataset_ref}")
    except NotFound:
        print(f"Creating dataset: {dataset_ref}...")
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"✓ Dataset created: {dataset_ref}")

    # 2. Create Table
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    try:
        table = client.get_table(table_ref)
        print(f"✓ Table exists: {table_ref}")
        
        # Check for missing columns and add them
        existing_cols = {f.name for f in table.schema}
        new_schema = list(table.schema)
        needs_update = False
        for field in schema:
            if field.name not in existing_cols:
                print(f"  Adding missing column: {field.name}")
                new_schema.append(field)
                needs_update = True
        
        if needs_update:
            table.schema = new_schema
            client.update_table(table, ["schema"])
            print("✓ Table schema updated")

    except NotFound:
        print(f"Creating table: {table_ref}...")
        table = bigquery.Table(table_ref, schema=schema)
        # Clustering for performance
        table.clustering_fields = ["source_url"]
        client.create_table(table)
        print(f"✓ Table created: {table_ref}")
        
        # Insert dummy row to allow index creation (BigQuery needs at least one row to calc stats)
        print("Inserting dummy row to initialize index...")
        dummy_embedding = [0.0] * 768  # Vertex AI text-embedding-005 dimension
        rows_to_insert = [{
            "chunk_id": "init_dummy",
            "content": "Initialization dummy row",
            "embedding": dummy_embedding,
            "source_url": "http://init",
            "metadata": "{}",
            "created_at": "2024-01-01T00:00:00"
        }]
        errors = client.insert_rows_json(table_ref, rows_to_insert)
        if errors:
            print(f"⚠ Warning: Could not insert dummy row: {errors}")
        else:
            print("✓ Dummy row inserted")

    # 3. Create Vector Index
    # Note: Vector indexes are created via SQL DDL
    index_name = f"{table_id}_embedding_idx"
    print(f"Checking/Creating vector index: {index_name}...")
    
    create_index_query = f"""
    CREATE VECTOR INDEX IF NOT EXISTS `{index_name}`
    ON `{table_ref}`(embedding)
    OPTIONS(distance_type='COSINE', index_type='IVF')
    """
    
    import time
    max_retries = 20
    for attempt in range(max_retries):
        try:
            job = client.query(create_index_query)
            job.result() # Wait for completion
            print(f"✓ Vector index ensured: {index_name}")
            break
        except Exception as e:
            if "Failed to calculate array_min_len" in str(e) or "all NULLs" in str(e):
                print(f"  (Attempt {attempt+1}/{max_retries}) Waiting for data to become available for indexing...")
                time.sleep(10) # Wait 10s before retry
            else:
                print(f"⚠ Warning creating vector index: {e}")
                print("  (You might need to wait a few minutes after table creation before creating the index)")
                break

    print("\nInitialization complete!")
    return True

if __name__ == "__main__":
    # Ensure we have credentials
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.getenv("BQ_PROJECT"):
        print("Warning: BQ_PROJECT not set. Using default 'cal-rag-agent'.")
        print("If this is incorrect, run: export BQ_PROJECT=your-project-id")
    
    try:
        init_bigquery()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
