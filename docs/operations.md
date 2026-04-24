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

When `.env` is copied from `.env.example`, RabbitMQ uses `RABBITMQ_DEFAULT_USER` and `RABBITMQ_DEFAULT_PASS` from that file. The host publisher should use a `localhost` `RABBITMQ_URL`; the Compose worker is wired to the `rabbitmq` service DNS name.

## Job Queue

Direct publisher and worker commands require `RABBITMQ_URL`. Copy `.env.example` to `.env` for local defaults, export the variable in your shell, or pass `--rabbitmq-url` to the publisher.

Publisher:

```bash
.venv/bin/python -m ingestion_pipeline.publisher --video /data/videos/example.mp4
```

Worker:

```bash
.venv/bin/python -m ingestion_pipeline.worker
```

The local worker imports the ingestion pipeline when it receives a job, so install `requirements-ingestion.txt` before running it outside Docker.

Messages are JSON objects with `video_path`, optional `output_dir`, optional `title`, and optional `year`. The worker acknowledges successful jobs and rejects failed jobs without requeueing, so failed jobs should be republished after fixing the underlying issue.

`SPEAKER_MAP_TIMEOUT_SECONDS` caps how long the worker waits for manual speaker identification output. The Compose and Kubernetes defaults use 3600 seconds so a missing `speaker_map.json` fails the job instead of occupying the worker forever.

Kubernetes queue components:

```bash
kubectl -n video-se rollout status deployment/rabbitmq
kubectl -n video-se rollout status deployment/ingestion-worker
kubectl -n video-se logs deployment/ingestion-worker -f
```

The bundled `k8s/rabbitmq.yaml` uses the same `video-se-secrets` object as the worker. For a managed broker, keep `RABBITMQ_URL` in that secret pointed at the managed endpoint and skip applying the bundled RabbitMQ manifest.

The default deploy path is `kubectl apply -k k8s/`. Apply `k8s/ingestion-job.yaml` only when you want a one-off ingestion job for a specific video path.

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
- Check `SPEAKER_MAP_TIMEOUT_SECONDS` if the worker rejected a job while waiting.
- Run the speaker UI and save the map, or provide the expected file under the processed video directory.

RabbitMQ worker does nothing:

- Check the queue name in `INGESTION_QUEUE`.
- Confirm the published `video_path` is valid inside the worker container.
- Check RabbitMQ management UI for ready/unacked messages.
