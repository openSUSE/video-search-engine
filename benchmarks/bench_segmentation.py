"""Benchmarks for the boundary-scoring segmentation step.

The CPU-bound work in ``ingestion_pipeline.steps.step_02_segmentation`` lives
in three places we want to track over time:

* ``_perform_boundary_scoring`` — the O(N) loop that computes per-shot
  boundary scores from text/visual/speaker/audio/action signals.
* ``_cosine_similarity`` and ``_vector_to_floats`` — the per-pair math that
  dominates the score loop.
* ``_validate_analysis_data`` — JSON shape validation invoked once per
  pipeline run on a potentially long shot list.

Embeddings are mocked with a deterministic fake model that returns
length-aware vectors based on the input text so that consecutive shots
produce distinct similarity values without paying the cost of a real
sentence-transformer.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from ingestion_pipeline.steps import step_02_segmentation as seg

from .harness import Benchmark


_DIM = 16


def _hashed_vector(text: str) -> List[float]:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=_DIM).digest()
    # Map bytes to floats in [-1, 1] so cosine similarity is non-trivial.
    return [(byte - 128) / 128.0 for byte in digest]


class _DeterministicEmbedder:
    """Embedding model that hashes input text into a stable vector."""

    def encode(self, sentences: List[str], **_: Any) -> List[List[float]]:
        return [_hashed_vector(text) for text in sentences]


def _make_rich_shots(count: int) -> List[Dict[str, Any]]:
    """Build a synthetic ``rich_shots`` list resembling pipeline output."""

    shots: List[Dict[str, Any]] = []
    for index in range(count):
        # Alternate speaker sets and audio/action mixes so the boundary loop
        # actually exercises every change signal.
        speaker_pool = ["Alice", "Bob", "Carol"]
        speakers = [speaker_pool[index % 3], speaker_pool[(index + 1) % 3]]
        shots.append(
            {
                "shot_id": f"shot_{index:04d}",
                "start_time": float(index) * 3.0,
                "end_time": float(index) * 3.0 + 2.5,
                "dialogue_text": f"Speaker line {index} about topic {index % 7}.",
                "visual_caption": f"A person standing in scene {index % 5}.",
                "speakers": sorted(speakers),
                "audio_events": ["speech", "music"] if index % 2 else ["speech"],
                "actions": ["talk", "walk"] if index % 3 == 0 else ["talk"],
            }
        )
    return shots


def _make_analysis_data(count: int) -> List[Dict[str, Any]]:
    """Build a payload accepted by ``_validate_analysis_data``."""

    payload: List[Dict[str, Any]] = []
    for index in range(count):
        payload.append(
            {
                "shot_id": f"shot_{index:04d}",
                "time_start_sec": float(index) * 3.0,
                "time_end_sec": float(index) * 3.0 + 2.5,
                "visual_caption": f"Scene {index}",
                "transcript_segments": [
                    {"speaker": "SPEAKER_00", "text": "Hello there."},
                    {"speaker": "SPEAKER_01", "text": "General Kenobi."},
                ],
                "audio_events": [{"event": "speech"}, {"event": "music"}],
                "detected_actions": [{"action": "talk"}, {"action": "walk"}],
            }
        )
    return payload


def _setup_boundary_scoring():
    rich_shots = _make_rich_shots(80)
    embedder = _DeterministicEmbedder()
    return rich_shots, embedder


def _bench_boundary_scoring(payload):
    rich_shots, embedder = payload
    seg._perform_boundary_scoring(rich_shots, embedder)


def _setup_cosine_pair():
    return _hashed_vector("left side of the dialogue"), _hashed_vector(
        "right side of the visual"
    )


def _bench_cosine_similarity(payload):
    left, right = payload
    seg._cosine_similarity(left, right)


def _bench_vector_to_floats(payload):
    left, _ = payload
    seg._vector_to_floats(left)


def _setup_validate_analysis():
    return _make_analysis_data(150)


def _bench_validate_analysis(payload):
    seg._validate_analysis_data(payload)


def _setup_prepare_rich_shots():
    return (
        _make_analysis_data(150),
        {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
    )


def _bench_prepare_rich_shots(payload):
    analysis, speaker_map = payload
    seg._prepare_rich_shots(analysis, speaker_map)


BENCHMARKS = [
    Benchmark(
        name="seg._perform_boundary_scoring(80 shots)",
        category="ingestion",
        description=(
            "Boundary scoring loop over 80 shots with deterministic embedder; "
            "exercises text/visual/speaker/audio/action change signals plus "
            "segment merging."
        ),
        fn=_bench_boundary_scoring,
        setup=_setup_boundary_scoring,
        iterations=200,
    ),
    Benchmark(
        name="seg._cosine_similarity",
        category="ingestion",
        description="Cosine similarity over a 16-dim float pair (hot loop body).",
        fn=_bench_cosine_similarity,
        setup=_setup_cosine_pair,
        iterations=20_000,
    ),
    Benchmark(
        name="seg._vector_to_floats",
        category="ingestion",
        description="Defensive numeric coercion of an embedding vector.",
        fn=_bench_vector_to_floats,
        setup=_setup_cosine_pair,
        iterations=20_000,
    ),
    Benchmark(
        name="seg._validate_analysis_data(150 shots)",
        category="ingestion",
        description=(
            "Full structural validation of a 150-shot final_analysis.json "
            "payload — runs once per ingestion job."
        ),
        fn=_bench_validate_analysis,
        setup=_setup_validate_analysis,
        iterations=400,
    ),
    Benchmark(
        name="seg._prepare_rich_shots(150 shots)",
        category="ingestion",
        description=(
            "Speaker-name mapping and dialogue consolidation across 150 "
            "shots — runs once per ingestion job."
        ),
        fn=_bench_prepare_rich_shots,
        setup=_setup_prepare_rich_shots,
        iterations=1_000,
    ),
]
