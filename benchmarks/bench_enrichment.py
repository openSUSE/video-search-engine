"""Benchmarks for the LLM enrichment helpers in step 03.

The network-bound calls to Gemini and Ollama are out of scope (we never want
benchmarks to depend on external services), but the surrounding pure-Python
plumbing runs once per segment and is worth tracking:

* ``_validate_segments`` — validates the ``final_segments.json`` payload that
  enrichment reads on every retry.
* ``_safe_llm_updates`` — defensively normalises an LLM response.
* ``_normalize_llm_keywords`` — keyword cleanup applied to LLM output.
* ``_has_complete_enrichment`` — gating predicate that decides whether a
  segment can be skipped on rerun.

The benchmarks mirror realistic input shapes (titles, summaries, mixed-type
keywords) so timings reflect production behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ingestion_pipeline.steps import step_03_enrichment as enrich

from .harness import Benchmark


def _make_segments(count: int) -> List[Dict[str, Any]]:
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
                "full_transcript": (
                    "Alice: hello there. Bob: general kenobi. "
                    f"Alice: today we discuss topic {index}."
                ),
                "consolidated_visual_captions": [
                    "Two people in a studio.",
                    "A close-up shot of Alice.",
                ],
                "consolidated_audio_events": ["speech"],
                "consolidated_actions": ["talk"],
            }
        )
    return segments


def _setup_validate_segments():
    return _make_segments(100)


def _bench_validate_segments(payload):
    enrich._validate_segments(payload)


def _setup_llm_updates():
    return {
        "title": "  A descriptive segment title  ",
        "summary": "Two people discuss the topic at hand for a couple of minutes.",
        "keywords": ["topic", " discussion ", "people", None, "", "interview"],
    }


def _bench_safe_llm_updates(payload):
    enrich._safe_llm_updates(payload)


def _bench_normalize_keywords(payload):
    enrich._normalize_llm_keywords(payload["keywords"])


def _setup_has_complete_enrichment():
    return {
        "segment_id": "segment_0001",
        "title": "Title",
        "summary": "Summary text.",
        "keywords": ["one", "two", "three"],
    }


def _bench_has_complete_enrichment(payload):
    enrich._has_complete_enrichment(payload)


def _setup_resolve_provider():
    return ("gemini", None)


def _bench_resolve_llm_client(payload):
    provider, clients = payload
    enrich._resolve_llm_client(provider, clients)


BENCHMARKS = [
    Benchmark(
        name="enrich._validate_segments(100)",
        category="ingestion",
        description="Validate a 100-segment final_segments.json payload.",
        fn=_bench_validate_segments,
        setup=_setup_validate_segments,
        iterations=500,
    ),
    Benchmark(
        name="enrich._safe_llm_updates",
        category="ingestion",
        description="Normalise an LLM response payload with mixed-type keywords.",
        fn=_bench_safe_llm_updates,
        setup=_setup_llm_updates,
        iterations=20_000,
    ),
    Benchmark(
        name="enrich._normalize_llm_keywords",
        category="ingestion",
        description="Keyword list cleanup with whitespace and falsy entries.",
        fn=_bench_normalize_keywords,
        setup=_setup_llm_updates,
        iterations=50_000,
    ),
    Benchmark(
        name="enrich._has_complete_enrichment",
        category="ingestion",
        description="Skip-gate predicate evaluated for every segment on rerun.",
        fn=_bench_has_complete_enrichment,
        setup=_setup_has_complete_enrichment,
        iterations=50_000,
    ),
    Benchmark(
        name="enrich._resolve_llm_client",
        category="ingestion",
        description="Provider resolution from config (no clients dict).",
        fn=_bench_resolve_llm_client,
        setup=_setup_resolve_provider,
        iterations=50_000,
    ),
]
