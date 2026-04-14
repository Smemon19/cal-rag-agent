# syntax=docker/dockerfile:1

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
