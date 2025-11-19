FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl tesseract-ocr tesseract-ocr-eng \
  && rm -rf /var/lib/apt/lists/*
  
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --prefer-binary --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
