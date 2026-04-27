"""Benchmarks for the FastAPI search layer.

The FastAPI handler in ``api.main`` delegates the heavy work to three pieces
of pure-Python code that this module exercises in isolation:

* ``api.schemas`` — Pydantic v2 input/output validation.
* ``api.search_utils.text_metadata_by_segment_id`` — fan-out filter over the
  ``collection.get`` payload returned by ChromaDB.
* ``api.search_service.HybridSearchService`` — RRF fusion across two
  ``collection.query`` results plus a ``collection.get`` round-trip.

The ``HybridSearchService`` benchmark replaces the embedding model and vector
collection with deterministic in-memory fakes so the timings reflect *only*
the code the production server runs synchronously.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from api.schemas import SearchQuery, SearchResponse
from api.search_service import HybridSearchService
from api.search_utils import text_metadata_by_segment_id

from .harness import Benchmark


def _metadata(segment_id: str, *, video: str = "demo.mp4") -> Dict[str, Any]:
    return {
        "title": f"Segment {segment_id}",
        "summary": f"A summary of {segment_id} that mentions the topic.",
        "speakers": "Alice,Bob",
        "keywords": "topic,demo,segment",
        "actions": "talking",
        "start_time": 1.0,
        "end_time": 5.0,
        "video_filename": video,
        "type": "text",
    }


class _StaticEmbeddingModel:
    """Embedding model returning a fixed 16-dim vector."""

    _vector: List[float] = [0.1 * i for i in range(16)]

    def encode(self, query: str) -> List[float]:  # noqa: ARG002 - signature parity
        return self._vector


class _SyntheticCollection:
    """In-memory stand-in for a ChromaDB collection used by HybridSearchService.

    The collection mimics the two access patterns used by the search service:
    ``query`` (vector search) and ``get`` (metadata lookup by id). Returns are
    constructed at init time so that ``query``/``get`` are constant-time and
    do not skew timings with their own setup work.
    """

    def __init__(self, segment_count: int = 30) -> None:
        self._segment_ids = [f"segment_{idx:04d}" for idx in range(segment_count)]
        self._video = "demo.mp4"
        self._text_ids = [
            [f"{self._video}::{segment_id}_text" for segment_id in self._segment_ids]
        ]
        self._visual_ids = [
            # Reverse to ensure RRF actually fuses two distinct rankings.
            [
                f"{self._video}::{segment_id}_visual"
                for segment_id in reversed(self._segment_ids)
            ]
        ]
        self._metadata_by_doc_id = {
            f"{self._video}::{segment_id}_text": _metadata(segment_id)
            for segment_id in self._segment_ids
        }

    def query(self, *, query_embeddings: Any, n_results: int, where: Dict[str, Any]):
        if where.get("type") == "text" or (
            "$and" in where and where["$and"][0]["type"] == "text"
        ):
            return {"ids": [self._text_ids[0][:n_results]]}
        return {"ids": [self._visual_ids[0][:n_results]]}

    def get(self, *, ids: List[str], include: List[str]):  # noqa: ARG002 - parity
        return {
            "ids": list(ids),
            "metadatas": [self._metadata_by_doc_id[doc_id] for doc_id in ids],
        }


def _setup_search_service() -> HybridSearchService:
    return HybridSearchService(_StaticEmbeddingModel(), _SyntheticCollection(30))


def _bench_search_service(service: HybridSearchService) -> None:
    service.search("a topic that matters", top_k=5, video_filename="demo.mp4")


def _setup_text_metadata() -> Tuple[List[str], List[Dict[str, Any]]]:
    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for index in range(50):
        segment_id = f"segment_{index:04d}"
        ids.append(f"demo.mp4::{segment_id}_text")
        metadatas.append(_metadata(segment_id))
        # Inject a few documents with the wrong suffix to exercise the filter.
        ids.append(f"demo.mp4::{segment_id}_visual")
        metadatas.append(_metadata(segment_id))
    return ids, metadatas


def _bench_text_metadata(payload: Tuple[List[str], List[Dict[str, Any]]]) -> None:
    ids, metadatas = payload
    text_metadata_by_segment_id(ids, metadatas)


def _bench_search_query_validate(_: Any) -> None:
    SearchQuery(query="find the highlights", top_k=5, video_filename="demo.mp4")


def _setup_search_response_dump() -> SearchResponse:
    return SearchResponse(
        results=[
            {
                "id": f"demo.mp4::segment_{i:04d}",
                "score": 0.5 + 0.001 * i,
                "start_time": float(i),
                "end_time": float(i) + 5.0,
                "title": f"Segment {i}",
                "summary": "A representative summary of the segment under test.",
                "video_filename": "demo.mp4",
                "speakers": "Alice,Bob",
            }
            for i in range(20)
        ]
    )


def _bench_search_response_dump(response: SearchResponse) -> None:
    response.model_dump()


def _bench_search_response_dump_json(response: SearchResponse) -> None:
    response.model_dump_json()


BENCHMARKS = [
    Benchmark(
        name="api.HybridSearchService.search",
        category="api",
        description=(
            "Full hybrid search with RRF fusion across 2x 30-document rankings "
            "and metadata lookup, using deterministic fake collection."
        ),
        fn=_bench_search_service,
        setup=_setup_search_service,
        iterations=2_000,
    ),
    Benchmark(
        name="api.text_metadata_by_segment_id",
        category="api",
        description=(
            "Filter 100 mixed text/visual ChromaDB rows down to a "
            "segment-id-keyed metadata map."
        ),
        fn=_bench_text_metadata,
        setup=_setup_text_metadata,
        iterations=5_000,
        inner_loops=100,
    ),
    Benchmark(
        name="api.SearchQuery.validate",
        category="api",
        description="Pydantic v2 validation of an inbound /search payload.",
        fn=_bench_search_query_validate,
        iterations=20_000,
    ),
    Benchmark(
        name="api.SearchResponse.model_dump",
        category="api",
        description="Serialise a 20-item /search response to a Python dict.",
        fn=_bench_search_response_dump,
        setup=_setup_search_response_dump,
        iterations=5_000,
    ),
    Benchmark(
        name="api.SearchResponse.model_dump_json",
        category="api",
        description="Serialise a 20-item /search response to a JSON string.",
        fn=_bench_search_response_dump_json,
        setup=_setup_search_response_dump,
        iterations=5_000,
    ),
]
