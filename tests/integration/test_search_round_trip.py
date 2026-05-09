"""End-to-end round-trip against a real ChromaDB.

This is the smallest test that would have caught the kind of regression the
mock-only unit tests miss: a renamed metadata key, a wrong distance space,
or a doc-id format that quietly stops matching.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _build_segment(
    segment_id: str,
    title: str,
    summary: str,
    keywords: list[str],
    *,
    video_filename: str = "itest_video",
    start: float = 0.0,
    end: float = 5.0,
) -> dict:
    return {
        "segment_id": segment_id,
        "start_time": start,
        "end_time": end,
        "speakers": [],
        "consolidated_visual_captions": ["a calm scene"],
        "consolidated_actions": ["walking"],
        "consolidated_audio_events": ["music"],
        "title": title,
        "summary": summary,
        "keywords": keywords,
        "full_transcript": "",
    }


def _index_segments(collection, segments, embedding_model, video_filename: str) -> None:
    """Mirror the production indexing payload shape.

    Kept inline rather than importing run_indexing because we want this test
    to fail loudly if the production payload shape drifts away from what
    the search service expects — that's the whole point.
    """
    from ingestion_pipeline.steps.step_04_indexing import run_indexing

    import json
    import os
    import tempfile

    # run_indexing reads a path on disk, so write a temp file and feed it.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "enriched.json")
        with open(path, "w") as f:
            json.dump(segments, f)

        ok = run_indexing(
            enriched_segments_path=path,
            video_filename=video_filename,
            config={
                "models": {"embedding": {"name": "all-MiniLM-L6-v2"}},
                "general": {"device": "cpu"},
                "database": {"host": "x", "port": 0, "collection_name": "x"},
            },
            embedding_model=embedding_model,
            collection=collection,
        )
        assert ok, "run_indexing must succeed for a well-formed segment list"


@pytest.fixture(scope="module")
def embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def test_index_then_search_round_trip(chroma_collection, embedding_model):
    """Index two visibly-different segments; assert each query finds its segment."""
    from api.search_service import HybridSearchService

    segments = [
        _build_segment(
            "segment_0001",
            title="A man cooking food in a kitchen",
            summary="The chef chops vegetables and sears meat in a hot skillet.",
            keywords=["cooking", "kitchen", "chef", "skillet"],
            start=0.0,
            end=5.0,
        ),
        _build_segment(
            "segment_0002",
            title="A spaceship lands on a desert planet",
            summary="A small craft touches down amid swirling sand and rocky plateaus.",
            keywords=["spaceship", "desert", "landing", "sci-fi"],
            start=10.0,
            end=15.0,
        ),
    ]
    _index_segments(
        chroma_collection,
        segments,
        embedding_model,
        video_filename="itest_video",
    )

    service = HybridSearchService(embedding_model, chroma_collection)

    cooking_results = service.search("food being prepared in a kitchen", top_k=2)
    assert cooking_results, "non-empty result list"
    assert cooking_results[0]["id"] == "itest_video::segment_0001"

    spaceship_results = service.search("space craft on dusty planet", top_k=2)
    assert spaceship_results
    assert spaceship_results[0]["id"] == "itest_video::segment_0002"


def test_search_returns_empty_for_empty_collection(chroma_collection, embedding_model):
    """An empty collection must return [] from /search, not raise."""
    from api.search_service import HybridSearchService

    service = HybridSearchService(embedding_model, chroma_collection)
    assert service.search("anything", top_k=5) == []


def test_video_filename_filter_isolates_results(chroma_collection, embedding_model):
    """Filtering by video_filename must only return that video's segments."""
    from api.search_service import HybridSearchService

    _index_segments(
        chroma_collection,
        [
            _build_segment(
                "segment_0001",
                title="Movie A: cooking",
                summary="A cooks in a kitchen.",
                keywords=["cooking"],
            )
        ],
        embedding_model,
        video_filename="movie_a",
    )
    _index_segments(
        chroma_collection,
        [
            _build_segment(
                "segment_0001",
                title="Movie B: cooking",
                summary="B cooks in a kitchen.",
                keywords=["cooking"],
            )
        ],
        embedding_model,
        video_filename="movie_b",
    )

    service = HybridSearchService(embedding_model, chroma_collection)
    results = service.search("cooking", top_k=5, video_filename="movie_a")
    assert results, "filter for movie_a should match its segment"
    assert all(r["video_filename"] == "movie_a" for r in results)
