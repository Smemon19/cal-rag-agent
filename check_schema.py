
import os
from google.cloud import bigquery
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

project = os.getenv("BQ_PROJECT")
dataset = os.getenv("BQ_DATASET")
table = os.getenv("BQ_TABLE")

print(f"Project: {project}")
print(f"Dataset: {dataset}")
print(f"Table: {table}")

if not all([project, dataset, table]):
    print("Error: Missing BQ environment variables.")
    exit(1)

try:
    client = bigquery.Client(project=project)
    table_ref = f"{project}.{dataset}.{table}"
    table_obj = client.get_table(table_ref)

    print(f"\nSchema for {table_ref}:")
    for schema_field in table_obj.schema:
        print(f"- {schema_field.name} ({schema_field.field_type})")

except Exception as e:
    print(f"Error inspecting schema: {e}")

