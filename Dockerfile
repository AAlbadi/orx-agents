# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    KOKORO_CACHE_DIR=/root/.cache/pipecat/kokoro-onnx \
    WHISPER_CACHE_DIR=/root/.cache/whisper

# Install essential system dependencies (espeak-ng for Kokoro, ffmpeg/libsndfile for audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy model pre-fetch script and download models during build to ensure instant container startup
COPY scripts/download_models.py scripts/
RUN python scripts/download_models.py

# Copy application source code
COPY . .

# Expose FastAPI and WebSocket port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
