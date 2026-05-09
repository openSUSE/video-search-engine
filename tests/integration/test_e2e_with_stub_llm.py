"""Deterministic E2E test: enrichment-with-stub-LLM → indexing → search.

Skips the heavy ML extraction stages (whisperx, AST, BLIP, VideoMAE) by
feeding pre-baked segment data, so this test is reproducible in CI without
requiring HF tokens or model downloads. Still talks to a real Chroma so
metadata-shape regressions in run_indexing get caught.

The fixture's job is to lock the contract between: enriched segment shape
→ what gets indexed → what /search returns. If anyone reshuffles metadata
keys without updating both sides, this test fails noisily.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

pytestmark = pytest.mark.integration


# Minimal pre-segmentation output: 2 segments, hand-tuned so a query about
# "rabbit chasing butterflies" deterministically picks segment_0001 and a
# query about "explosion" picks segment_0002. The contents are intentionally
# distinct enough that even a small embedding model gets the ranking right.
SAMPLE_SEGMENTS = [
    {
        "segment_id": "segment_0001",
        "segment_index": 1,
        "start_time": 0.0,
        "end_time": 6.0,
        "duration_sec": 6.0,
        "speakers": [],
        "full_transcript": "",
        "consolidated_visual_captions": [
            "a rabbit running through a sunny meadow",
            "small white butterflies fluttering near flowers",
        ],
        "consolidated_actions": ["running", "playing", "leaping"],
        "consolidated_audio_events": ["birds chirping", "wind"],
    },
    {
        "segment_id": "segment_0002",
        "segment_index": 2,
        "start_time": 6.0,
        "end_time": 12.0,
        "duration_sec": 6.0,
        "speakers": [],
        "full_transcript": "",
        "consolidated_visual_captions": [
            "a fiery explosion in a dark cave",
            "smoke and falling debris",
        ],
        "consolidated_actions": ["explosion", "running away"],
        "consolidated_audio_events": ["loud bang", "rumble", "crackling fire"],
    },
]


def _stub_gemini_client(prompt, _config):
    """Return deterministic enrichment based on which segment is in the prompt.

    The real provider would call Gemini; in tests we route off cues that
    appear in the prompt template. Using prompt content (not segment id)
    means we exercise the same _render_prompt → llm_client path the
    production code takes.
    """
    if "rabbit" in prompt.lower():
        return {
            "title": "Rabbit chasing butterflies in a meadow",
            "summary": "A rabbit playfully runs through a sunny meadow, chasing white butterflies near wildflowers.",
            "keywords": ["rabbit", "meadow", "butterflies", "sunny", "playful"],
        }
    if "explosion" in prompt.lower():
        return {
            "title": "Cave explosion with fire and debris",
            "summary": "A fiery explosion erupts in a dark cave, sending smoke, debris, and embers flying.",
            "keywords": ["explosion", "cave", "fire", "debris", "smoke"],
        }
    return None


@pytest.fixture(scope="module")
def embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def test_enrichment_indexing_search_round_trip(chroma_collection, embedding_model):
    from api.search_service import HybridSearchService
    from ingestion_pipeline.steps.step_03_enrichment import run_enrichment
    from ingestion_pipeline.steps.step_04_indexing import run_indexing

    config = {
        "filenames": {"enriched_segments": "final_enriched_segments.json"},
        "llm_enrichment": {"provider": "gemini"},
        "models": {"embedding": {"name": "all-MiniLM-L6-v2"}},
        "general": {"device": "cpu"},
        "database": {
            "host": "x",
            "port": 0,
            "collection_name": "x",  # injected via fixture below
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        segments_path = os.path.join(tmp, "final_segments.json")
        with open(segments_path, "w") as f:
            json.dump(SAMPLE_SEGMENTS, f)

        # Step 3: enrichment with our stub Gemini client.
        enriched_path = run_enrichment(
            segments_path,
            config,
            llm_clients={"gemini": _stub_gemini_client},
        )
        assert enriched_path, "enrichment must succeed with the stub client"

        # The enriched file should contain title/summary/keywords for both.
        with open(enriched_path) as f:
            enriched = json.load(f)
        assert enriched[0]["title"] == "Rabbit chasing butterflies in a meadow"
        assert enriched[1]["title"] == "Cave explosion with fire and debris"

        # Step 4: indexing into the integration-test Chroma collection.
        ok = run_indexing(
            enriched_segments_path=enriched_path,
            video_filename="e2e_fixture",
            config=config,
            embedding_model=embedding_model,
            collection=chroma_collection,
        )
        assert ok, "run_indexing must succeed for stub-enriched segments"

    # Search through the public service; assert deterministic top hits.
    service = HybridSearchService(embedding_model, chroma_collection)

    rabbit_hits = service.search("rabbit chasing butterflies in meadow", top_k=2)
    assert rabbit_hits, "search must return at least one result"
    assert rabbit_hits[0]["id"] == "e2e_fixture::segment_0001"
    assert rabbit_hits[0]["video_filename"] == "e2e_fixture"

    explosion_hits = service.search("loud explosion in a cave", top_k=2)
    assert explosion_hits
    assert explosion_hits[0]["id"] == "e2e_fixture::segment_0002"
