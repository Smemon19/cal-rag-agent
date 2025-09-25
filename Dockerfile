# syntax=docker/dockerfile:1

FROM python:3.12-slim

# System dependencies for Tesseract OCR and common libs used by imaging
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tesseract-ocr \
       libtesseract-dev \
       libleptonica-dev \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app

# Cloud Run will send traffic to this port
EXPOSE 8080

# Start Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]


