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

- API liveness: `http://localhost:1234/healthz` — returns 200 once the FastAPI process is up.
- API readiness: `http://localhost:1234/readyz` — returns 503 until the embedding model and ChromaDB connection are initialized; use this for load-balancer or Kubernetes readiness probes.
- Demo UI: `http://localhost:8501` — multi-page Streamlit (Home / Submit / Pipeline / Search). The Pipeline page polls `data/processed/<video>/` every 1.5 s for live step-by-step progress.
- Speaker identification UI: `http://localhost:5050`
- RabbitMQ management: `http://localhost:15672`
- ChromaDB: `http://localhost:8000`

For a full end-to-end demo, also start the worker so the Submit page has somewhere to send the job:

```bash
make compose-worker
```

Then on the Submit page, pick a video already staged under `data/videos/`, optionally set title/year, and click **Publish ingestion job**. The Pipeline page picks up the job automatically and animates the per-step progress (shot detection → audio → transcription → audio events → visual captions → actions → speaker map → segmentation → enrichment → indexing).

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

## Logging

`core.logger.setup_logging` honors `LOG_LEVEL` (CRITICAL, ERROR, WARNING, INFO, DEBUG). Default is INFO; unknown values fall back to INFO. Set `LOG_LEVEL=DEBUG` in the API/worker env to surface deeper diagnostics without redeploying code. Every long-running entrypoint — the FastAPI service, the ingestion worker, the Flask speaker UI, and the publisher CLI — calls `setup_logging` at boot, so the variable applies uniformly; configure it via `LOG_LEVEL` in `.env` for Compose or via `video-se-config` in Kubernetes.

`LOG_FORMAT` selects the output format:

- `pretty` (default): colored text suited for local development. Each line includes `req=<id>` so you can grep for a specific request.
- `json`: one JSON object per line with `timestamp`, `level`, `logger`, `message`, `request_id`, plus any caller-supplied `extra=` fields. Recommended for production deployments so a log shipper (Loki, ELK, Datadog) can parse and index without a regex.

Every API request is bound to a request id via the contextvar in `core.observability`. Inbound `X-Request-ID` headers are honored; missing headers get a fresh 12-char hex id. The id is echoed in the response header and surfaces on every log line emitted during the request, so a failed `/search` can be traced from the access log to the search service to ChromaDB without grepping by timestamp.

## Security

The API ships open by default — appropriate for a development stack on a
trusted network — and exposes three opt-in knobs. Each is independent;
enable any combination by setting the corresponding env var.

| Env var | Effect |
|---|---|
| `SEARCH_API_KEY` | When set, `/search` requires `X-API-Key: <value>`. Mismatch returns `401` with `WWW-Authenticate: X-API-Key`. `/healthz`, `/readyz`, `/metrics` stay open so probes and scrapers don't need the key. |
| `CORS_ALLOW_ORIGINS` | Comma-separated origin allow-list (e.g. `https://app.example.com,https://staging.example.com`). When unset, no CORS middleware is attached at all (cross-origin browser calls are simply not possible). |
| `MAX_REQUEST_BODY_BYTES` | Hard cap on `Content-Length` for state-changing requests. Default 32 KiB. Bodies larger than this are rejected with `413` before the handler allocates. |
| `RATE_LIMIT_PER_MINUTE` | Per-client `/search` budget over a 60-second sliding window. `0` / unset disables. Bucketing prefers `X-API-Key` (one budget per credential), then `X-Forwarded-For` first hop, then peer IP. Exceeding the budget returns `429` with `Retry-After: <seconds>`. `/healthz`, `/readyz`, `/metrics` are exempt so probes/scrapers can't be locked out. |

Compose users set these in `.env`; Kubernetes users put `SEARCH_API_KEY`
in the `video-se-secrets` Secret and `CORS_ALLOW_ORIGINS` /
`MAX_REQUEST_BODY_BYTES` in the `video-se-config` ConfigMap.

The 401 response always includes the request id from the
observability middleware — `kubectl logs` filtered on `req=<id>` shows
exactly which auth attempt failed and why.

Every auth attempt (granted *and* denied) emits a structured audit
record on the `api.security.audit` logger:

```json
{
  "audit": "search_api_auth",
  "outcome": "denied",
  "method": "POST",
  "path": "/search",
  "peer_ip": "10.42.0.7",
  "forwarded_ip": "203.0.113.4",
  "key_presented": true,
  "key_prefix": "badk…",
  "request_id": "ca9c07fa367e",
  "level": "WARNING",
  ...
}
```

`key_prefix` is the first 4 characters of the presented key followed by
an ellipsis — never the full credential. Pair it with the
`request_id` and `peer_ip` to trace credential brute-forcing or post-leak
abuse. Audit denials log at WARNING; grants at INFO so production
deployments can dial down the volume by raising the level on
`api.security.audit` while still capturing failures.

## Metrics

The API exposes Prometheus metrics at `GET /metrics`:

| Metric | Type | Labels |
|---|---|---|
| `video_se_http_requests_total` | counter | `route`, `method`, `status` (`2xx`/`4xx`/`5xx`) |
| `video_se_http_request_duration_seconds` | histogram | `route`, `method` |
| `video_se_search_requests_total` | counter | `outcome` (`success`/`error`) |
| `video_se_search_results_count` | histogram | — |

Routes that aren't recognized templates collapse to `/_other` to keep label cardinality bounded. The `/metrics` endpoint itself is excluded from the counters so a scraper doesn't dominate its own dashboards.

`prometheus-client` is a runtime dep of `requirements-api.txt`. If it's missing (e.g. in a stripped-down image) the API still boots — `api.observability` logs a warning and the `/metrics` route 404s.

## Model And Data Volumes

The Compose stack uses persistent volumes for ChromaDB, RabbitMQ, and model caches. In Kubernetes, the manifests expect PVCs for videos, processed data, and model caches.

Service Dockerfiles, Chroma, and RabbitMQ are pinned to release images instead of floating tags. For local Compose, set `CHROMA_IMAGE_TAG` or `RABBITMQ_IMAGE_TAG` in `.env` when testing an upgrade; for Kubernetes, update `k8s/chroma.yaml` or `k8s/rabbitmq.yaml` and validate before rollout. For service image base upgrades, update the `FROM` line in each file under `docker/`.

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

Search latency / recall:

- Successful `/search` calls emit `Search returned <N> results in <ms>ms (top_k=..., video=...)`.
- Failures emit `Search failed after <ms>ms (top_k=..., video=...).` followed by the traceback. Both paths log duration, so a slow timeout vs. a fast crash is distinguishable in logs.
- The text-side embedding is built from `title + keywords + (transcript or summary)`, keeping short semantic anchors ahead of long transcripts. The visual/audio-side embedding is built from `summary + visual captions + actions + audio events`, so sound cues such as music or applause stay searchable even when the transcript is silent. Changes to those recipes in `step_04_indexing.py` only take effect after re-running the ingestion pipeline on each video — `collection.upsert` overwrites by id, so reingesting a video replaces its embeddings without dropping the collection.

Pipeline waits after extraction:

- `SPEAKER_UI_MODE=external` means the pipeline waits for `speaker_map.json`.
- Check `SPEAKER_MAP_TIMEOUT_SECONDS` if the worker rejected a job while waiting.
- Run the speaker UI and save the map, or provide the expected file under the processed video directory.

RabbitMQ worker does nothing:

- Check the queue name in `INGESTION_QUEUE`.
- Confirm the published `video_path` is valid inside the worker container.
- Check RabbitMQ management UI for ready/unacked messages.

## Dead-Letter Queue

Failed jobs are routed to `video.ingestion.dlq` via the
`video.ingestion.dlx` direct exchange (routing key `video.ingestion.failed`).
The worker `basic_nack(requeue=False)`s on any unhandled exception, so the
broker moves the message to the DLQ instead of dropping it. Inspect with the
RabbitMQ management UI at `http://localhost:15672` or via `rabbitmqadmin`:

```bash
rabbitmqadmin -u "$RABBITMQ_DEFAULT_USER" -p "$RABBITMQ_DEFAULT_PASS" \
  list queues name messages_ready messages_unacknowledged
rabbitmqadmin -u "$RABBITMQ_DEFAULT_USER" -p "$RABBITMQ_DEFAULT_PASS" \
  get queue=video.ingestion.dlq count=5 ackmode=ack_requeue_false
```

Republish a fixed job after correcting the underlying problem; nothing
auto-replays from the DLQ.

If `_open_channel` raises with "queue exists with different arguments", an
older queue declaration without the DLX argument is colliding with the new
topology. Either delete the queue and let it be recreated, or set
`INGESTION_QUEUE` to a versioned name (e.g. `video.ingestion.v2`) and
redeploy.

## Search API Filters

`POST /search` accepts these optional fields beyond `query` and `top_k`:

- `video_filename`: restrict results to one video stem (e.g. `sintel_trailer`).
- `min_duration_sec` / `max_duration_sec`: drop segments outside the range
  *after* hybrid fusion. Useful when very short cut-segments dominate the
  top hits for noisy collections.

Indexing-time metadata filters are now exposed on every Chroma row:

- `speakers_tokens`, `keywords_tokens`, `actions_tokens`,
  `audio_events_tokens` — pipe-delimited lowercase strings (`|alice|bob|`).
  Use these for precise matching with Chroma's `$contains` operator, e.g.
  `where: {"speakers_tokens": {"$contains": "|sintel|"}}`. The
  comma-joined display fields (`speakers`, `keywords`, ...) match only the
  whole joined string and shouldn't be used for filtering.
- `duration_sec` — numeric, useful for `$gte` / `$lte` server-side filters
  before fetching metadata.
- `context_hash` — sha256 over `(embedding_model_name, indexed_text)`. The
  indexer reads these on rerun and short-circuits when every segment's hash
  matches what's stored, skipping the encode + upsert entirely.

## Cache Freshness Sidecars

Each model-bound extraction output (`transcript_raw.json`,
`audio_events.json`, `visual_details.json`, `actions.json`) is paired with
a `<output>.cache_meta.json` sidecar that records the model name and key
parameters. On rerun the pipeline reuses the cached output only when the
sidecar matches the current `config.yaml`. Switching models in
`config.yaml` therefore invalidates just the affected step rather than
silently reusing stale data.

Legacy artifacts produced before the sidecar was added are accepted on
first read (the sidecar is backfilled with the current expected meta);
afterwards, normal mismatch invalidation applies.

## Dependency Audits

`make audit` runs `pip-audit` against the pinned requirement files
(`requirements-api.txt`, `requirements-dev.txt`, `requirements-ui.txt`)
and exits non-zero if any package version has a known CVE in the PyPI
advisory database. The `dependency-audit` CI lane runs the same scan on
every push and PR.

The heavy ingestion deps (`torch`, `whisperx`, etc.) are intentionally
excluded — their advisories are noisy and the worker is not directly
internet-facing in our deployment shape. To audit them too, override the
file list:

```bash
make audit AUDIT_FILES="requirements-api.txt requirements-dev.txt requirements-ui.txt requirements-ingestion.txt"
```

When a CVE fires, fix it by bumping the affected package in the
requirements file, re-running `make audit` to confirm clean, and
re-running `make bench-check` to confirm no perf regression.

## Tests And Benchmarks

Unit suite (no external services):

```bash
make test
```

Integration suite (requires a running compose stack — `make compose-up` first):

```bash
make test-integration
# or override the host explicitly:
CHROMA_HOST=127.0.0.1 CHROMA_PORT=8000 make test-integration
```

Integration tests in `tests/integration/` skip themselves when Chroma is
unreachable, so the same files work locally and in the `compose-smoke` CI
job.

Benchmark suite (deterministic in-process; no external services):

```bash
make bench-smoke         # 10% iterations, fast sanity check
make bench               # full suite
make bench-baseline      # write benchmarks/reports/baseline.json
make bench-check         # compare current code vs baseline; fails on >10% regression
```
