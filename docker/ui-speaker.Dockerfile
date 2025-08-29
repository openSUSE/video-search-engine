FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt pandas

COPY . .

EXPOSE 5050
CMD ["streamlit","run","app/ui/speaker_id_tool.py","--server.address","0.0.0.0","--server.port","5050"]
