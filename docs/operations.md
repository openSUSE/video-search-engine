# Operations Runbook

## Local Services

Start the core stack:

```bash
make compose-up
```

Start the RabbitMQ ingestion worker:

```bash
make compose-worker
```

Start the speaker identification UI:

```bash
make compose-speaker
```

Stop the stack:

```bash
make compose-down
```

Useful endpoints:

- API health: `http://localhost:1234/healthz`
- Search UI: `http://localhost:8501`
- Speaker identification UI: `http://localhost:5050`
- RabbitMQ management: `http://localhost:15672`
- ChromaDB: `http://localhost:8000`

## Job Queue

Publisher:

```bash
.venv/bin/python -m ingestion_pipeline.publisher --video /data/videos/example.mp4
```

Worker:

```bash
RABBITMQ_URL=amqp://guest:guest@localhost:5672/%2F \
.venv/bin/python -m ingestion_pipeline.worker
```

The local worker imports the ingestion pipeline when it receives a job, so install `requirements-ingestion.txt` before running it outside Docker.

Messages are JSON objects with `video_path`, optional `output_dir`, optional `title`, and optional `year`. The worker acknowledges successful jobs and rejects failed jobs without requeueing, so failed jobs should be republished after fixing the underlying issue.

## Secrets

Required secrets are environment variables:

- `HF_TOKEN`
- `GEMINI_API_KEY` when using Gemini
- `TMDB_API_KEY` when metadata lookup is desired

Do not add these to tracked YAML. Use `.env` locally, Docker/Kubernetes secrets in hosted environments, and `config.example.yaml` for non-secret defaults.

## Model And Data Volumes

The Compose stack uses persistent volumes for ChromaDB, RabbitMQ, and model caches. In Kubernetes, the manifests expect PVCs for videos, processed data, and model caches.

Large model downloads happen on first run. Keep `MODEL_CACHE_DIR` or `/models` persistent between restarts to avoid repeated downloads.

## Troubleshooting

Port already in use:

```bash
lsof -nP -iTCP:1234 -iTCP:8501 -iTCP:8000 -iTCP:5672 -iTCP:15672
```

Missing FFmpeg:

```bash
sudo apt update && sudo apt install ffmpeg
```

API starts but returns no results:

- Confirm ingestion completed.
- Confirm `CHROMA_COLLECTION` is the same for ingestion and API.
- Confirm the UI sends the selected video's stem as `video_filename`.

Pipeline waits after extraction:

- `SPEAKER_UI_MODE=external` means the pipeline waits for `speaker_map.json`.
- Run the speaker UI and save the map, or provide the expected file under the processed video directory.

RabbitMQ worker does nothing:

- Check the queue name in `INGESTION_QUEUE`.
- Confirm the published `video_path` is valid inside the worker container.
- Check RabbitMQ management UI for ready/unacked messages.
