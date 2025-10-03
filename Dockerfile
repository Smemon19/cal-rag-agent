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
# Upgrade pip tooling and preinstall binary wheels for numpy/pandas to avoid build toolchains
RUN pip install --upgrade pip setuptools wheel \
    && pip install --only-binary=:all: numpy==1.26.4 pandas==2.2.2 \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app

# Cloud Run will send traffic to this port
EXPOSE 8080

# Start Streamlit app (headless, bind to 0.0.0.0 and honor PORT)
ENV STREAMLIT_SERVER_HEADLESS=true
CMD ["sh", "-c", "exec streamlit run streamlit_app.py --server.port ${PORT:-8080} --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false"]


