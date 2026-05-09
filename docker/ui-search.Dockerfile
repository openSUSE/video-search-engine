FROM python:3.12.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt

COPY . .

EXPOSE 8501
# --server.fileWatcherType=none avoids Streamlit hitting the inotify
# instance limit on container startup (the default 'auto' walks every
# imported module + chromadb/transformers ship a lot of them). Hot-
# reload isn't useful in a deployed image anyway.
CMD ["streamlit","run","app/ui/search_app.py","--server.address","0.0.0.0","--server.port","8501","--server.fileWatcherType","none","--server.headless","true"]
