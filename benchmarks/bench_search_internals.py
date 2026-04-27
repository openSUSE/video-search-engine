"""Benchmarks for hot internals of ``api.search_service``.

The end-to-end ``HybridSearchService.search`` benchmark in ``bench_api`` is
useful for comparing against itself, but a regression in any of the small
helpers below is hard to localise from the aggregate. These per-helper
benchmarks make it cheap to spot which layer slowed down.
"""

from __future__ import annotations

from typing import Any, Dict, List

from api import search_service

from .harness import Benchmark


_RRF_K = 60


def _setup_query_results():
    return {
        "ids": [
            [f"demo.mp4::segment_{i:04d}_text" for i in range(60)]
        ]
    }


def _bench_first_id_list(results: Dict[str, Any]) -> None:
    search_service._first_id_list(results)


def _bench_query_segment_ids(results: Dict[str, Any]) -> None:
    search_service._query_segment_ids(results, "_text")


def _bench_where_clause(_: Any) -> None:
    search_service._where_clause("text", "demo.mp4")
    search_service._where_clause("visual", None)


def _setup_format_metadata():
    return {
        "title": "Title",
        "summary": "Summary text",
        "video_filename": "demo.mp4",
        "speakers": "Alice,Bob",
        "start_time": 1.0,
        "end_time": 5.0,
    }


def _bench_format_search_result(metadata: Dict[str, Any]) -> None:
    search_service._format_search_result("segment_0001", 0.5, metadata)


def _setup_query_vector_pair():
    class _Model:
        def encode(self, query: str) -> List[float]:  # noqa: ARG002 - signature parity
            return [0.01 * i for i in range(384)]

    return _Model()


def _bench_query_vector(model: Any) -> None:
    search_service._query_vector(model, "find the highlights")


def _setup_rrf_fusion():
    text_ids = [f"segment_{i:04d}" for i in range(60)]
    visual_ids = list(reversed(text_ids))
    return text_ids, visual_ids


def _bench_rrf_fusion(payload) -> None:
    text_ids, visual_ids = payload
    fused: Dict[str, float] = {}
    for rank, segment_id in enumerate(text_ids):
        fused[segment_id] = fused.get(segment_id, 0) + 1 / (rank + _RRF_K)
    for rank, segment_id in enumerate(visual_ids):
        fused[segment_id] = fused.get(segment_id, 0) + 1 / (rank + _RRF_K)
    sorted(fused, key=lambda segment_id: fused[segment_id], reverse=True)[:10]


BENCHMARKS = [
    Benchmark(
        name="search_service._first_id_list",
        category="api",
        description="Defensive first-row extraction from a query() result.",
        fn=_bench_first_id_list,
        setup=_setup_query_results,
        iterations=50_000,
    ),
    Benchmark(
        name="search_service._query_segment_ids(60)",
        category="api",
        description="Suffix-strip 60 query result ids into segment ids.",
        fn=_bench_query_segment_ids,
        setup=_setup_query_results,
        iterations=10_000,
        inner_loops=60,
    ),
    Benchmark(
        name="search_service._where_clause.x2",
        category="api",
        description="Two _where_clause builds (with/without video filter).",
        fn=_bench_where_clause,
        iterations=50_000,
        inner_loops=2,
    ),
    Benchmark(
        name="search_service._format_search_result",
        category="api",
        description="Build a single SearchResult dict from raw metadata.",
        fn=_bench_format_search_result,
        setup=_setup_format_metadata,
        iterations=50_000,
    ),
    Benchmark(
        name="search_service._query_vector(384)",
        category="api",
        description=(
            "Validate and coerce a 384-dim model.encode() output (matches "
            "all-MiniLM-L6-v2 dimensions)."
        ),
        fn=_bench_query_vector,
        setup=_setup_query_vector_pair,
        iterations=20_000,
    ),
    Benchmark(
        name="search_service.rrf_fusion(60+60)",
        category="api",
        description=(
            "Reciprocal Rank Fusion over two 60-document rankings, mirroring "
            "the inline loop in HybridSearchService.search."
        ),
        fn=_bench_rrf_fusion,
        setup=_setup_rrf_fusion,
        iterations=2_000,
        inner_loops=60,
    ),
]
