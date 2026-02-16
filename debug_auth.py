import os
import sys
from dotenv import load_dotenv

# Add current dir to path
sys.path.append(os.getcwd())

# Load .env to simulate user environment
load_dotenv(override=True)

print(f"GOOGLE_APPLICATION_CREDENTIALS='{os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}'")

try:
    from utils_bigquery import get_bq_client
    client = get_bq_client()
    print("Success: Client created via utils_bigquery")
    print(f"Project: {client.project}")
except Exception as e:
    print(f"Error type: {type(e)}")
    print(f"Error message: {e}")
    import traceback
    traceback.print_exc()
