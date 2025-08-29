FROM python:3.12-slim

# System libs: ffmpeg for audio; libsndfile1 for librosa; libglib for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl libsndfile1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Model caches on a writable path (mount a PVC here in k8s)
ENV HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    SENTENCE_TRANSFORMERS_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Only copy reqs first for better layer caching
COPY requirements-ingestion.txt .
RUN pip install --no-cache-dir -r requirements-ingestion.txt

# Copy the rest
COPY . .

# Default command runs the whole pipeline; k8s Job will override args
CMD ["python","-m","ingestion_pipeline.run_pipeline","--help"]
