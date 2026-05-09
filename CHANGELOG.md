# Changelog

All user-facing changes since the last shipped revision. Sections are
grouped by theme, not by chronological date — reviewers can read this in
the order they care about.

## Correctness

- **Pinned every runtime dependency.** `whisperx` is now pinned to a
  known-good commit; `torch`, `transformers`, `chromadb`, `pydantic`,
  `sentence-transformers`, `fastapi`, etc. all have explicit versions in
  `requirements-*.txt`. A fresh container build is now reproducible. The
  unpinned `whisperx` previously caused a hard crash when its diarization
  API changed underfoot.
- **Pyannote diarization API call updated** to the post-3.x signature
  (`token=` instead of `use_auth_token=`). The old code crashed on the
  current whisperx/pyannote combination.
- **`chromadb` is now in `requirements-ingestion.txt`.** The worker image
  previously had no chromadb, so step 4 of the pipeline (indexing) couldn't
  run inside the container at all.
- **Silent / no-audio video clips no longer crash the pipeline.** A pre-
  flight `ffprobe` detects the missing audio stream and writes empty
  transcript / audio-event sidecars, letting downstream steps proceed.
- **Enrichment prompts survive `{` and `}` in transcripts** (code tutorials,
  JSON quotes, math). The previous `prompt.format(...)` raised KeyError on
  any stray brace.
- **Per-segment enrichment failures no longer halt a whole video.** Up to
  20% of segments may stay errored on a transient LLM outage; the rest
  index and the failing IDs are logged for retry.
- **TMDb `metadata_fetcher` no longer swallows every exception** as
  "movie not found". Narrowed to `requests.RequestException` /
  `TMDbException`; one retry on rate limit.
- **Chroma `hnsw:space` is now asserted** in one authoritative place
  (`core/chroma_setup.py`). A collection that already exists with a
  different distance function raises loudly instead of silently degrading
  ranking.
- **`docker-compose.yml` chroma healthcheck** rewritten — the previous
  `python -c ...` healthcheck failed forever because the chroma image has
  no `python` binary; now uses `bash </dev/tcp/127.0.0.1/8000`.

## Search quality

- **Embeddings are normalized at both index and query time**
  (`normalize_embeddings=True`). Cosine ranking is now consistent and
  Chroma skips internal renormalization.
- **Hybrid candidate pool widened to `max(top_k * 5, 50)` per modality**
  before RRF fusion. Many co-occurring text+visual hits used to be dropped
  before reranking.
- **Boundary scoring handles silent shots correctly.** Empty-text
  embeddings used to give nonsense `text_change` values; now silent-vs-
  silent → 0 and silent-vs-talking → 1 explicitly.
- **VideoMAE handles short shots** by sampling with replacement instead of
  dropping them with `actions: []`.
- **AST runs on sliding 10-second windows** for shots longer than its
  training context, then aggregates label scores by mean.
- **BLIP `model.generate` runs in `torch.inference_mode()`** — autograd
  state is no longer accumulated per shot.
- **Audio is decoded once** and shared between whisperx and AST instead of
  decoded twice.

## Search API surface

- **`POST /search`** now accepts optional `min_duration_sec` /
  `max_duration_sec` filters that drop segments outside the range after
  hybrid fusion.
- **Indexed metadata** now includes `<field>_tokens` pipe-delimited
  lowercase strings (`speakers_tokens`, `keywords_tokens`,
  `actions_tokens`, `audio_events_tokens`) for precise `$contains`
  filtering, plus `duration_sec` for numeric range filters and
  `context_hash` for caching.
- **Re-running indexing on unchanged segments is now a no-op.** The
  `context_hash` short-circuit skips encode + upsert when every segment's
  hash matches what's stored.

## Cache freshness

- **Per-step config-hash sidecars** (`<output>.cache_meta.json`) record
  which model produced each cached extraction artifact. Switching models
  in `config.yaml` invalidates exactly the affected step instead of
  silently reusing stale outputs.

## Reliability

- **Gemini calls retry** with bounded jittered exponential backoff (3
  attempts max). A transient 429/500 no longer marks the segment errored.
- **`subprocess.run(ffmpeg, ...)`** has a 30-minute timeout. A pathological
  input video can no longer hang the worker indefinitely.
- **`api/search_service.create_search_service` has bounded retries** (6
  attempts, exponential backoff) on the Chroma connect + heartbeat probe
  rather than a single unbounded blocking call.
- **RabbitMQ DLX/DLQ topology** declared automatically on first connect.
  Failed jobs (`basic_nack(requeue=False)`) land in `video.ingestion.dlq`
  for inspection instead of vanishing.
- **`queue_declare` mismatch** produces an actionable runtime error
  pointing at the deletion + recreate flow instead of a bare 406.

## Operability

- **Lazy `CONFIG`.** `from core.config import CONFIG` no longer triggers
  `.env` parsing + `os.environ` mutation at import time. Test collection
  is faster and side-effect-free.
- **Removed deprecated `TRANSFORMERS_CACHE`** from Dockerfiles and
  `core/config.py`; only `HF_HOME` is set now.
- **Robust MPS device probe** — wraps `torch.backends.mps` lookups in
  try/except and requires both `is_built()` and `is_available()`.
- **MIME-types widened in the search UI** dropdown to include `.mkv`,
  `.webm`, `.m4v`, `.mpg`, `.mpeg`. Previously the worker could ingest
  these but the UI silently filtered them out.

## Observability

- **`X-Request-ID` middleware.** Inbound header is honored when present;
  otherwise a fresh 12-char hex id is minted. Echoed in the response
  header. Available everywhere via `core.observability.current_request_id`.
- **`LOG_FORMAT=json|pretty` env var.** JSON mode emits one parseable
  record per line including `request_id`, `level`, `logger`, `message`,
  and any `extra=` fields. Pretty mode (default) is the colorlog path
  with `req=<id>` in the line.
- **`GET /metrics` Prometheus endpoint.** Counters and histograms for
  `video_se_http_requests_total`, `video_se_http_request_duration_seconds`,
  `video_se_search_requests_total`, `video_se_search_results_count`. Routes
  collapse to `/_other` for unrecognized paths to bound label cardinality.

## Security

All four knobs are opt-in (env-driven) and default to the previous
permissive behavior so existing deployments are unaffected.

- **`SEARCH_API_KEY`** — when set, `/search` requires `X-API-Key: <value>`
  and returns 401 with `WWW-Authenticate: X-API-Key` on mismatch.
  `/healthz`, `/readyz`, `/metrics` stay open for probes/scrapers.
- **`CORS_ALLOW_ORIGINS`** — comma-separated allow-list. Unset means no
  CORS middleware is mounted at all (cross-origin browser calls are
  simply not possible).
- **`MAX_REQUEST_BODY_BYTES`** — default 32 KiB. Oversized requests
  short-circuit with 413 before the handler allocates.
- **`RATE_LIMIT_PER_MINUTE`** — per-client sliding-window limiter (60-second
  window). Returns 429 with `Retry-After: <seconds>`. Bucketing keys: API
  key, then `X-Forwarded-For` first hop, then peer IP.
- **Auth audit logging** — every auth attempt (granted + denied) emits a
  structured record on the `api.security.audit` logger including
  `request_id`, `peer_ip`, `forwarded_ip`, `key_presented` (bool), and
  `key_prefix` (first 4 chars + ellipsis — *never* the full key). With
  `LOG_FORMAT=json`, this is directly indexable by a log shipper. Set the
  log level for that logger to dial volume up or down without changing the
  app default.

## Tests, CI, benchmarks

- **910 → 920 unit tests.** New: 11 observability tests, 27 security
  tests, plus updated coverage for the indexing metadata schema.
- **`tests/integration/`** runs against a live Chroma. Skips itself when
  Chroma is unreachable so the same files work locally and in CI.
  Includes a deterministic E2E with a stub Gemini client that asserts
  `enrichment → indexing → search` returns the expected segment id.
- **`compose-smoke` CI lane** brings up the compose stack and asserts
  `/healthz`, `/readyz`, `/search`, and the integration test pass against
  it. Would have caught the chroma healthcheck regression that escaped
  `make validate`.
- **`make test-integration`** target separates the integration suite from
  the unit suite. `make test` skips integration with `-m "not integration"`.
- **48 benchmarks** with regression gating: `make bench-baseline` to
  capture, `make bench-check` to compare. `WARN_RATIO` defaults to 0.15
  to suppress sub-microsecond CPU jitter while still catching real
  regressions.
- **`make audit`** runs `pip-audit` against the pinned api/dev/ui
  requirement sets and fails on any known CVE. Mirrored as a
  `dependency-audit` CI lane that runs on every push/PR. Today's
  baseline is clean (0 vulnerabilities) across all three sets.

## Documentation

- **`README.md`** — new "Search API" section listing all query fields,
  including the new `min/max_duration_sec` and the `_tokens` filter recipes.
- **`docs/operations.md`** — new sections on the DLQ, cache freshness
  sidecars, security knobs, structured logging, the `/metrics` endpoint,
  and the integration / benchmark test workflows.
- **`.env.example`** — every new env var is listed with a one-line
  comment explaining what it does and what unset means.
