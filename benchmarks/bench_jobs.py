"""Benchmarks for ingestion job message handling.

The publisher and worker exchange compact JSON payloads on a RabbitMQ queue.
Encode and decode happen on every job, so we want to keep both fast and to
catch regressions when the dataclass shape is extended. The benchmarks here
exercise the publisher -> wire -> worker round-trip purely in-process.
"""

from __future__ import annotations

from ingestion_pipeline.jobs import (
    IngestionJob,
    decode_job_message,
    encode_job_message,
)

from .harness import Benchmark


def _setup_full_job() -> IngestionJob:
    return IngestionJob(
        video_path="/data/videos/sample.mp4",
        output_dir="/data/processed",
        title="Sample",
        year=2024,
    )


def _setup_minimal_job() -> IngestionJob:
    return IngestionJob(video_path="/data/videos/sample.mp4")


def _setup_encoded_full() -> bytes:
    return encode_job_message(_setup_full_job()).encode("utf-8")


def _bench_encode(job: IngestionJob) -> None:
    encode_job_message(job)


def _bench_decode(body: bytes) -> None:
    decode_job_message(body)


def _bench_round_trip(job: IngestionJob) -> None:
    decode_job_message(encode_job_message(job).encode("utf-8"))


BENCHMARKS = [
    Benchmark(
        name="jobs.encode_job_message(full)",
        category="jobs",
        description="JSON-encode a fully populated IngestionJob.",
        fn=_bench_encode,
        setup=_setup_full_job,
        iterations=20_000,
    ),
    Benchmark(
        name="jobs.encode_job_message(minimal)",
        category="jobs",
        description="JSON-encode an IngestionJob with only video_path.",
        fn=_bench_encode,
        setup=_setup_minimal_job,
        iterations=20_000,
    ),
    Benchmark(
        name="jobs.decode_job_message",
        category="jobs",
        description="Decode and validate a wire-format ingestion job message.",
        fn=_bench_decode,
        setup=_setup_encoded_full,
        iterations=20_000,
    ),
    Benchmark(
        name="jobs.encode_decode_round_trip",
        category="jobs",
        description="Encode + decode a full job, end-to-end in-process.",
        fn=_bench_round_trip,
        setup=_setup_full_job,
        iterations=10_000,
    ),
]
