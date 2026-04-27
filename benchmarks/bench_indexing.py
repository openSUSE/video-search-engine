"""Benchmarks for the ChromaDB indexing helpers (step 04).

Step 04 runs once per ingestion job, but its CPU footprint scales with the
segment count and is dominated by three pure-Python helpers that this module
benchmarks in isolation:

* ``_validate_segments`` — defensive shape check over the enriched segments
  list before they are pushed to ChromaDB.
* ``_prepare_metadata_for_db`` — build the per-document metadata dict that
  ChromaDB requires (string-only values, joined keyword lists, etc.).
* ``_encoded_vectors_to_lists`` — coerce a sentence-transformer output (or a
  list-of-lists fake) into the validated ``List[List[float]]`` format the
  ``upsert`` call expects.

Embeddings are simulated with deterministic float lists rather than real
torch tensors so that the benchmark runs without GPU/CPU model dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ingestion_pipeline.steps import step_04_indexing as indexing

from .harness import Benchmark


def _make_enriched_segments(count: int) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for index in range(count):
        segments.append(
            {
                "segment_id": f"segment_{index:04d}",
                "segment_index": index + 1,
                "start_time": float(index) * 4.0,
                "end_time": float(index) * 4.0 + 3.5,
                "duration_sec": 3.5,
                "speakers": ["Alice", "Bob"],
                "full_transcript": f"Transcript text for segment {index}.",
                "consolidated_visual_captions": [
                    "Two people in a studio.",
                    "Close-up of speaker.",
                ],
                "consolidated_audio_events": ["speech"],
                "consolidated_actions": ["talk"],
                "title": f"Title {index}",
                "summary": f"Summary for segment {index}.",
                "keywords": ["topic", "segment", f"k{index}"],
            }
        )
    return segments


def _setup_validate_segments():
    return _make_enriched_segments(100)


def _bench_validate_segments(payload):
    indexing._validate_segments(payload)


def _setup_prepare_metadata():
    return _make_enriched_segments(1)[0]


def _bench_prepare_metadata(segment):
    indexing._prepare_metadata_for_db(segment, "demo.mp4")


def _setup_encoded_vectors():
    # 100 segments * 16-dim, list of lists — mimics a SentenceTransformer
    # ``encode`` result on CPU when ``convert_to_numpy=True`` and ``.tolist()``
    # has been applied (or a fake model used in tests).
    return [
        [0.001 * i + 0.01 * j for j in range(16)]
        for i in range(100)
    ]


def _bench_encoded_vectors(vectors):
    indexing._encoded_vectors_to_lists(vectors, len(vectors), "text")


def _bench_join_metadata_values(_: Any) -> None:
    indexing._join_metadata_values(["one", "two", "three", "four", "five"])
    indexing._join_metadata_values("already-joined")
    indexing._join_metadata_values([])


BENCHMARKS = [
    Benchmark(
        name="indexing._validate_segments(100)",
        category="ingestion",
        description="Validate a 100-segment final_enriched_segments.json payload.",
        fn=_bench_validate_segments,
        setup=_setup_validate_segments,
        iterations=500,
    ),
    Benchmark(
        name="indexing._prepare_metadata_for_db",
        category="ingestion",
        description="Build a single ChromaDB-compatible metadata dict per segment.",
        fn=_bench_prepare_metadata,
        setup=_setup_prepare_metadata,
        iterations=50_000,
    ),
    Benchmark(
        name="indexing._encoded_vectors_to_lists(100x16)",
        category="ingestion",
        description=(
            "Coerce a 100x16 embedding batch into validated List[List[float]]. "
            "Bounds-checks dimensions and rejects bool/non-finite entries."
        ),
        fn=_bench_encoded_vectors,
        setup=_setup_encoded_vectors,
        iterations=400,
        inner_loops=100,
    ),
    Benchmark(
        name="indexing._join_metadata_values.x3",
        category="ingestion",
        description="Three calls to _join_metadata_values covering list/str/empty.",
        fn=_bench_join_metadata_values,
        iterations=50_000,
        inner_loops=3,
    ),
]
