import json

import pytest

from ingestion_pipeline.steps import step_04_indexing as indexing_step
from ingestion_pipeline.steps.step_04_indexing import run_indexing


class FakeEmbeddingModel:
    def __init__(self):
        self.encode_calls = []

    def encode(self, texts, show_progress_bar):
        self.encode_calls.append(
            {
                "texts": texts,
                "show_progress_bar": show_progress_bar,
            }
        )
        call_index = len(self.encode_calls) - 1
        return [[call_index, text_index] for text_index, _text in enumerate(texts)]


class FakeCollection:
    def __init__(self):
        self.upsert_call = None

    def upsert(self, *, ids, embeddings, metadatas):
        self.upsert_call = {
            "ids": ids,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }


def test_run_indexing_builds_text_and_visual_entries_with_injected_dependencies(tmp_path):
    enriched_segments_path = tmp_path / "segments.json"
    enriched_segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment-1",
                    "full_transcript": "spoken words",
                    "summary": "person enters",
                    "speakers": ["Alice", "Bob"],
                    "keywords": ["arrival"],
                    "consolidated_actions": ["walking"],
                    "consolidated_visual_captions": ["a doorway"],
                    "start_time": 1.5,
                    "end_time": 4.0,
                },
                {
                    "segment_id": "segment-2",
                    "full_transcript": "",
                    "summary": "fallback summary",
                    "start_time": 5,
                    "end_time": 8,
                },
            ]
        )
    )
    embedding_model = FakeEmbeddingModel()
    collection = FakeCollection()

    run_indexing(
        str(enriched_segments_path),
        "demo-video",
        {"database": {"collection_name": "unused"}},
        embedding_model=embedding_model,
        collection=collection,
    )

    assert embedding_model.encode_calls == [
        {
            "texts": ["spoken words", "fallback summary"],
            "show_progress_bar": True,
        },
        {
            "texts": [
                "person enters. a doorway. walking",
                "fallback summary",
            ],
            "show_progress_bar": True,
        },
    ]
    assert collection.upsert_call["ids"] == [
        "demo-video::segment-1_text",
        "demo-video::segment-1_visual",
        "demo-video::segment-2_text",
        "demo-video::segment-2_visual",
    ]
    assert collection.upsert_call["embeddings"] == [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ]
    assert collection.upsert_call["metadatas"] == [
        {
            "title": "",
            "summary": "person enters",
            "speakers": "Alice,Bob",
            "keywords": "arrival",
            "actions": "walking",
            "start_time": 1.5,
            "end_time": 4.0,
            "video_filename": "demo-video",
            "type": "text",
        },
        {
            "title": "",
            "summary": "person enters",
            "speakers": "Alice,Bob",
            "keywords": "arrival",
            "actions": "walking",
            "start_time": 1.5,
            "end_time": 4.0,
            "video_filename": "demo-video",
            "type": "visual",
        },
        {
            "title": "",
            "summary": "fallback summary",
            "speakers": "",
            "keywords": "",
            "actions": "",
            "start_time": 5.0,
            "end_time": 8.0,
            "video_filename": "demo-video",
            "type": "text",
        },
        {
            "title": "",
            "summary": "fallback summary",
            "speakers": "",
            "keywords": "",
            "actions": "",
            "start_time": 5.0,
            "end_time": 8.0,
            "video_filename": "demo-video",
            "type": "visual",
        },
    ]


def test_run_indexing_skips_empty_segment_files_without_creating_dependencies(tmp_path):
    enriched_segments_path = tmp_path / "segments.json"
    enriched_segments_path.write_text("[]")

    run_indexing(
        str(enriched_segments_path),
        "demo-video",
        {},
    )


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ({"segment_id": "segment-1"}, "JSON array"),
        (["not a segment"], "index 0"),
        ([{"summary": "missing id"}], "segment_id"),
        ([{"segment_id": " "}], "segment_id"),
    ],
)
def test_run_indexing_rejects_invalid_segments_before_creating_dependencies(
    monkeypatch,
    tmp_path,
    segments,
    message,
):
    enriched_segments_path = tmp_path / "segments.json"
    enriched_segments_path.write_text(json.dumps(segments))

    def fail_create_dependency(_config):
        raise AssertionError("indexing dependencies should not load for invalid segment data")

    monkeypatch.setattr(indexing_step, "create_embedding_model", fail_create_dependency)
    monkeypatch.setattr(indexing_step, "create_vector_collection", fail_create_dependency)

    with pytest.raises(ValueError, match=message):
        run_indexing(
            str(enriched_segments_path),
            "demo-video",
            {"database": {"collection_name": "unused"}},
        )
