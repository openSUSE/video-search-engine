FROM python:3.12-slim

# Small set of libs for OpenCV wheels & networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    SENTENCE_TRANSFORMERS_HOME=/models/hf \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt .
# Install CPU torch wheel explicitly so sentence-transformers stays light
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt

COPY . .

EXPOSE 1234
# k8s Deployment will just run this
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","1234"]
