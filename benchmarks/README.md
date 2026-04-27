# Benchmark Suite

Microbenchmarks for the hot paths of the video search engine. The suite is
implemented on top of the standard library only (no `pytest-benchmark`,
`pyperf`, or third-party runners) so it has the same dependency footprint as
the existing test suite and runs in any environment that already supports
`make test`.

## What Is Covered

The runner discovers benchmarks from the modules listed in
`benchmarks.runner.BENCHMARK_MODULES`. The current coverage:

| Category    | What it measures |
|-------------|------------------|
| `config`    | YAML parsing, `core.config.load_config`, device selection, `setup_logging` first-call vs idempotent paths, `Logger.info` dispatch cost. |
| `api`       | End-to-end `HybridSearchService.search`, RRF fusion in isolation, query-vector validation, where-clause builder, metadata fan-out, result formatting, Pydantic v2 schema validation, response serialisation. |
| `ingestion` | Boundary scoring loop, cosine similarity, validation of `final_analysis.json` / `final_segments.json` / `final_enriched_segments.json`, LLM-output normalisation, ChromaDB metadata prep, embedding-batch coercion. |
| `jobs`      | Encode/decode of RabbitMQ ingestion job messages. |
| `ui`        | Streamlit search-client serialisation, transcript validation, speaker-map normalisation, URL building. |

Heavy external pieces (`sentence-transformers`, ChromaDB, FastAPI server,
RabbitMQ, Gemini/Ollama, FFmpeg, WhisperX) are intentionally out of scope —
benchmarks must not depend on external services or GPU hardware. Wherever
those collaborators sit on the hot path we use deterministic in-process
fakes (see `benchmarks/bench_api.py::_StaticEmbeddingModel` and
`benchmarks/bench_segmentation.py::_DeterministicEmbedder`).

## Running

```bash
make bench                         # full suite
make bench-smoke                   # 10% iterations, quiet output (CI-friendly)

# Direct invocations
.venv/bin/python -m benchmarks.runner
.venv/bin/python -m benchmarks.runner --filter '^api\.'
.venv/bin/python -m benchmarks.runner --scale 0.25 --json bench.json
.venv/bin/python -m benchmarks.runner --format md > bench.md
.venv/bin/python -m benchmarks.runner --baseline previous.json
.venv/bin/python -m benchmarks.runner --baseline previous.json --fail-on-regression
.venv/bin/python -m benchmarks.runner --list
```

`--filter` accepts a regular expression and may be passed multiple times.
Each pattern is matched against the benchmark name *and* the category
independently, so anchors like `^api\.` Just Work without needing to know
the internal `category::name` representation.

`--scale` multiplies the iteration count of every benchmark uniformly.
Use values below 1.0 for smoke runs (`make bench-smoke` uses 0.1) and values
above 1.0 for offline measurement when investigating a regression.

`--json PATH` writes a deterministic JSON document with metadata
(interpreter, platform, scale) and per-benchmark statistics (`min`,
`median`, `mean`, `p95`, `p99`, `max`, `stdev`, `total`, `ops_per_second`).
The shape is suitable for diffing across runs and for feeding into trend
dashboards.

`--format {text,md,json}` controls the stdout report. `text` is the default
fixed-width table; `md` emits a GitHub-flavoured markdown table suitable for
pasting into PR comments or README snippets; `json` mirrors the `--json`
artifact on stdout.

`--baseline PATH` loads a previous `--json` snapshot and prints a
median-vs-median comparison table on stderr after the run. Combined with
`--fail-on-regression` and a `--warn-ratio` threshold (default `0.10`,
i.e. 10% of the baseline median) the runner exits with status 1 if any
benchmark slowed down past the threshold — useful for blocking PRs on
known-slow paths. Median is the comparison axis because it ignores the
tail, which is dominated by environmental noise rather than code changes.

## Reading The Output

Each benchmark reports nanosecond-scale statistics:

* `min`, `median`, `mean` — central tendency. The median is robust to
  jitter; the mean is what `ops_per_second` is derived from.
* `p95`, `p99` — tail latency. Wide tails usually indicate GC, allocator
  noise, or unrelated CPU contention rather than code regressions.
* `stdev` — sample standard deviation. Treat numbers above ~25% of the
  mean as noisy; rerun at a higher `--scale` to confirm.
* `ops` — derived from the mean and `inner_loops`.

Benchmarks declaring `inner_loops` (for example
`api.text_metadata_by_segment_id` with `inner_loops=100`) divide every
sample by the inner-loop count, so the reported timings are *per logical
operation*, not per call.

## Adding a Benchmark

1. Create or extend a module under `benchmarks/`. Each module exports a
   `BENCHMARKS` list of `benchmarks.harness.Benchmark` instances.
2. Pick an `iterations` count that keeps the run under ~250ms total wall
   time on a developer laptop while staying above ~100ns per iteration to
   avoid clock resolution bias. Use `inner_loops` when the callable
   performs N units of work per call to keep iteration counts manageable.
3. Use the `setup` callable to build expensive fixtures once. The harness
   passes its return value to `fn` and never re-runs `setup` between
   timed iterations.
4. Add the module to `BENCHMARK_MODULES` in `benchmarks/runner.py`.
5. Run `make bench-smoke` to confirm discovery, naming, and timings look
   sensible.

The runner enforces unique benchmark names across modules. Categories are
free-form labels — pick an existing category when extending coverage of an
existing area (`api`, `ingestion`, `ui`, `jobs`, `config`) or introduce a
new one when adding a new subsystem.

## Stability Tips

* Close anything CPU-heavy before running `make bench`. The harness
  disables Python's GC during timing windows but cannot defend against
  external contention.
* On laptops, plug in the charger first — battery saver modes throttle
  the CPU and skew comparisons.
* Compare runs at the same `--scale`. Doubling iterations does not
  change the per-op mean meaningfully but does shrink the confidence
  interval, which can make a small regression look like noise.
