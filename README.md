# Semantic Video Search Engine

An end-to-end video ingestion and semantic search system. The pipeline extracts transcript, speaker, visual, action, and audio-event metadata from videos, enriches segments with an LLM, indexes them in ChromaDB, and serves hybrid text/visual search through FastAPI and Streamlit.

## What Is Included

- FastAPI search API backed by ChromaDB vector search.
- Streamlit search UI for selecting videos and jumping to result timestamps.
- Batch ingestion pipeline for extraction, segmentation, enrichment, and indexing.
- RabbitMQ ingestion queue with publisher and worker entrypoints.
- Docker Compose stack for local API, UI, ChromaDB, RabbitMQ, and optional worker.
- Kubernetes manifests for production-style deployment.
- Lightweight tests, Compose validation, and CI.

## Architecture

```text
data/videos/*.mp4
  -> ingestion_pipeline.run_pipeline
  -> extracted transcript, shots, visual captions, actions, audio events
  -> manual speaker map
  -> enriched segments
  -> ChromaDB collection
  -> FastAPI search API
  -> Streamlit search UI
```

RabbitMQ can decouple ingestion from callers:

```text
publisher CLI -> RabbitMQ queue -> ingestion worker -> pipeline -> ChromaDB
```

## Prerequisites

- Python 3.12 recommended.
- Docker and Docker Compose.
- FFmpeg for local ingestion outside Docker.
- Enough disk for model caches and processed media.
- `HF_TOKEN` for WhisperX speaker diarization.
- `GEMINI_API_KEY` when `LLM_PROVIDER=gemini`.
- Optional `TMDB_API_KEY` for movie metadata lookup.

## Local Setup

```bash
cp .env.example .env
make install-dev
make validate
```

Edit `.env` and set real secrets. Tracked config files are safe defaults only; credentials must come from environment variables.

For local ingestion outside Docker, install the heavier runtime dependencies into the same virtualenv:

```bash
.venv/bin/python -m pip install -r requirements-ingestion.txt -r requirements-ui.txt
```

## Run The Stack

Start ChromaDB, RabbitMQ, the API, and the search UI:

```bash
make compose-up
```

Default URLs:

- Search UI: `http://localhost:8501`
- Speaker identification UI: `http://localhost:5050` when started with `make compose-speaker`
- Search API: `http://localhost:1234`
- API health: `http://localhost:1234/healthz`
- RabbitMQ management: `http://localhost:15672` with credentials from `.env` or `video_se` / `video_se_dev` if unset
- ChromaDB: `http://localhost:8000`

If you copied `.env.example`, RabbitMQ uses the local development credentials from that file. The container worker connects through the Compose service name, while host-side publisher commands use `localhost`.

Stop services:

```bash
make compose-down
```

## Ingest A Video

Put videos under `data/videos`.

Run ingestion synchronously from the local virtualenv:

```bash
.venv/bin/python -m ingestion_pipeline.run_pipeline --video data/videos/your_video.mp4
```

Optional metadata:

```bash
.venv/bin/python -m ingestion_pipeline.run_pipeline \
  --video data/videos/your_video.mp4 \
  --title "Movie Title" \
  --year 2024
```

By default `SPEAKER_UI_MODE=external`, so the pipeline waits for the configured `speaker_map.json` after extraction. Set `SPEAKER_MAP_TIMEOUT_SECONDS` to cap that wait for workers, or leave it empty for no timeout. Run the speaker identification UI in another terminal when needed:

```bash
.venv/bin/streamlit run app/ui/speaker_id_tool.py
```

Or run the containerized speaker UI against the Compose data mounts:

```bash
make compose-speaker
```

## Queue-Based Ingestion

Start the worker profile:

```bash
make compose-worker
```

Publish a job. When targeting the container worker, use the path as seen inside the worker container:

```bash
make publish-ingest VIDEO=/data/videos/your_video.mp4
```

Equivalent direct command:

```bash
.venv/bin/python -m ingestion_pipeline.publisher --video /data/videos/your_video.mp4
```

The worker consumes `INGESTION_QUEUE` from `RABBITMQ_URL`, runs the same pipeline, acknowledges successful jobs, and rejects failed jobs without requeueing.

## Configuration

Config is loaded from `CONFIG_PATH` or `config.yaml`, then environment variables override runtime values.

Important variables:

- `HF_TOKEN`, `GEMINI_API_KEY`, `TMDB_API_KEY`
- `ML_DEVICE`, `OUTPUT_DIR`, `VIDEO_DATA_PATH`, `MODEL_CACHE_DIR`
- `SPEAKER_UI_MODE`, `SPEAKER_MAP_TIMEOUT_SECONDS`
- `API_HOST`, `API_PORT`, `UI_HOST`, `UI_PORT`
- `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_COLLECTION`
- `RABBITMQ_URL`, `INGESTION_QUEUE`
- `RABBITMQ_DEFAULT_USER`, `RABBITMQ_DEFAULT_PASS` for local Compose or the bundled Kubernetes RabbitMQ
- `LLM_PROVIDER`, `GEMINI_MODEL`, `OLLAMA_HOST`, `OLLAMA_PORT`, `OLLAMA_MODEL`

Use `config.example.yaml` and `.env.example` as references. Do not put secrets in tracked YAML.

## Development

Run the lightweight validation suite:

```bash
make validate
```

This runs unit tests, validates Compose config with and without the worker profile, and compiles lightweight Python entrypoints. CI runs the same target.

## Deployment

Build and publish the Docker images from `docker/`:

- `docker/api.Dockerfile`
- `docker/ingestion.Dockerfile`
- `docker/ui-search.Dockerfile`
- `docker/ui-speaker.Dockerfile`

Kubernetes manifests live in `k8s/`. Create secrets outside Git:

```bash
kubectl apply -f k8s/ns.yaml
kubectl -n video-se create secret generic video-se-secrets \
  --from-literal=HF_TOKEN=... \
  --from-literal=GEMINI_API_KEY=... \
  --from-literal=TMDB_API_KEY=... \
  --from-literal=RABBITMQ_DEFAULT_USER=video_se \
  --from-literal=RABBITMQ_DEFAULT_PASS='change-me' \
  --from-literal=RABBITMQ_URL='amqp://video_se:change-me@rabbitmq:5672/%2F'
kubectl apply -k k8s/
```

Replace `<REG>/video-se-*:TAG` image placeholders in the manifests with your registry and immutable image tags. The bundled kustomization includes RabbitMQ and an `ingestion-worker` deployment; set `RABBITMQ_URL` to a managed broker endpoint instead if you do not want to run RabbitMQ in-cluster. `k8s/ingestion-job.yaml` and `k8s/toolbox.yaml` are manual operational helpers and are intentionally excluded from the default kustomization.

## Operations

See [docs/operations.md](docs/operations.md) for local runbook commands, environment notes, queue behavior, and troubleshooting.
