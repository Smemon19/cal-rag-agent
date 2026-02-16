import os
import sys
from dotenv import load_dotenv

# Add current dir to path
sys.path.append(os.getcwd())

# Load .env
load_dotenv(override=True)

print("Attempting to create embedding function without chromadb...")

try:
    from utils import create_embedding_function
    fn = create_embedding_function()
    print("Success: Embedding function created")
    
    # Test with a dummy input
    embeddings = fn(["test"])
    print(f"Success: Generated embedding of length {len(embeddings[0])}")

except ImportError as e:
    if "chromadb" in str(e):
        print("FAIL: Still trying to import chromadb!")
        print(e)
    else:
        print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
