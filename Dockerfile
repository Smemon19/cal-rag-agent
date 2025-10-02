# syntax=docker/dockerfile:1

FROM python:3.11-slim

# System dependencies for compatibility and imaging
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl git ca-certificates \
       tesseract-ocr \
       libtesseract-dev \
       libleptonica-dev \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_HOME=/app/models \
    SENTENCE_TRANSFORMERS_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image to avoid runtime downloads
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
import os
model_name = "sentence-transformers/all-MiniLM-L6-v2"
target_dir = os.environ.get("HF_HOME", "/app/models")
os.makedirs(target_dir, exist_ok=True)
# Download to cache at target_dir by setting envs above
_ = SentenceTransformer(model_name)
print("Model baked at:", target_dir)
PY

# Copy the rest of the app
COPY . /app

# Cloud Run will send traffic to this port
EXPOSE 8080

# Start Streamlit app (headless, bind to 0.0.0.0:8080)
ENV STREAMLIT_SERVER_HEADLESS=true
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]


