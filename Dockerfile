# Mixr backend (FastAPI). The frontend deploys separately (Vercel, root: frontend/).
FROM python:3.12-slim

# libsndfile: soundfile I/O; ffmpeg: MP3 decode fallback; rubberband-cli: time-stretching.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg rubberband-cli \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    MIXR_STORAGE_DIR=/app/backend/storage

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
RUN mkdir -p "$MIXR_STORAGE_DIR"

EXPOSE 8000

# One worker only: tracks, trials and mixes live in process memory.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
