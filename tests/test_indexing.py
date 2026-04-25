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


def test_run_indexing_normalizes_video_filename_before_building_document_ids(tmp_path):
    enriched_segments_path = tmp_path / "segments.json"
    enriched_segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment-1",
                    "summary": "person enters",
                    "start_time": 1.5,
                    "end_time": 4.0,
                }
            ]
        )
    )
    collection = FakeCollection()

    run_indexing(
        str(enriched_segments_path),
        "  demo-video  ",
        {"database": {"collection_name": "unused"}},
        embedding_model=FakeEmbeddingModel(),
        collection=collection,
    )

    assert collection.upsert_call["ids"] == [
        "demo-video::segment-1_text",
        "demo-video::segment-1_visual",
    ]


@pytest.mark.parametrize(
    ("bad_call_index", "message"),
    [
        (0, "text embedding count"),
        (1, "visual embedding count"),
    ],
)
def test_run_indexing_rejects_embedding_count_mismatch_before_upsert(
    tmp_path,
    bad_call_index,
    message,
):
    enriched_segments_path = tmp_path / "segments.json"
    enriched_segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment-1",
                    "summary": "person enters",
                    "start_time": 1.5,
                    "end_time": 4.0,
                }
            ]
        )
    )
    collection = FakeCollection()

    class ShortEmbeddingModel:
        def __init__(self):
            self.call_count = 0

        def encode(self, texts, show_progress_bar):
            call_index = self.call_count
            self.call_count += 1
            if call_index == bad_call_index:
                return []
            return [[call_index, text_index] for text_index, _text in enumerate(texts)]

    with pytest.raises(ValueError, match=message):
        run_indexing(
            str(enriched_segments_path),
            "demo-video",
            {"database": {"collection_name": "unused"}},
            embedding_model=ShortEmbeddingModel(),
            collection=collection,
        )

    assert collection.upsert_call is None


@pytest.mark.parametrize(
    ("bad_call_index", "message"),
    [
        (0, "text embeddings must have consistent dimensions"),
        (1, "visual embeddings must have consistent dimensions"),
    ],
)
def test_run_indexing_rejects_embedding_dimension_mismatch_before_upsert(
    tmp_path,
    bad_call_index,
    message,
):
    enriched_segments_path = tmp_path / "segments.json"
    enriched_segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment-1",
                    "summary": "person enters",
                    "start_time": 1.5,
                    "end_time": 4.0,
                },
                {
                    "segment_id": "segment-2",
                    "summary": "person exits",
                    "start_time": 5.0,
                    "end_time": 7.0,
                },
            ]
        )
    )
    collection = FakeCollection()

    class MismatchedEmbeddingModel:
        def __init__(self):
            self.call_count = 0

        def encode(self, texts, show_progress_bar):
            call_index = self.call_count
            self.call_count += 1
            if call_index == bad_call_index:
                return [[1.0, 0.0], [1.0, 0.0, 0.0]]
            return [[call_index, text_index] for text_index, _text in enumerate(texts)]

    with pytest.raises(ValueError, match=message):
        run_indexing(
            str(enriched_segments_path),
            "demo-video",
            {"database": {"collection_name": "unused"}},
            embedding_model=MismatchedEmbeddingModel(),
            collection=collection,
        )

    assert collection.upsert_call is None


@pytest.mark.parametrize(
    ("bad_call_index", "message"),
    [
        (0, "text embeddings must be numeric vectors"),
        (1, "visual embeddings must be numeric vectors"),
    ],
)
def test_run_indexing_rejects_nonnumeric_embeddings_before_upsert(
    tmp_path,
    bad_call_index,
    message,
):
    enriched_segments_path = tmp_path / "segments.json"
    enriched_segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment-1",
                    "summary": "person enters",
                    "start_time": 1.5,
                    "end_time": 4.0,
                }
            ]
        )
    )
    collection = FakeCollection()

    class NonnumericEmbeddingModel:
        def __init__(self):
            self.call_count = 0

        def encode(self, texts, show_progress_bar):
            call_index = self.call_count
            self.call_count += 1
            if call_index == bad_call_index:
                return [["not-a-number"] for _text in texts]
            return [[call_index, text_index] for text_index, _text in enumerate(texts)]

    with pytest.raises(ValueError, match=message):
        run_indexing(
            str(enriched_segments_path),
            "demo-video",
            {"database": {"collection_name": "unused"}},
            embedding_model=NonnumericEmbeddingModel(),
            collection=collection,
        )

    assert collection.upsert_call is None


def test_run_indexing_rejects_nonfinite_embeddings_before_upsert(tmp_path):
    enriched_segments_path = tmp_path / "segments.json"
    enriched_segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment-1",
                    "summary": "person enters",
                    "start_time": 1.5,
                    "end_time": 4.0,
                }
            ]
        )
    )
    collection = FakeCollection()

    class NonfiniteEmbeddingModel:
        def encode(self, texts, show_progress_bar):
            return [[float("nan")] for _text in texts]

    with pytest.raises(ValueError, match="text embeddings must be numeric vectors"):
        run_indexing(
            str(enriched_segments_path),
            "demo-video",
            {"database": {"collection_name": "unused"}},
            embedding_model=NonfiniteEmbeddingModel(),
            collection=collection,
        )

    assert collection.upsert_call is None


@pytest.mark.parametrize("video_filename", ["", "   ", None])
def test_run_indexing_rejects_invalid_video_filename_before_creating_dependencies(
    monkeypatch,
    tmp_path,
    video_filename,
):
    enriched_segments_path = tmp_path / "segments.json"
    enriched_segments_path.write_text(json.dumps([{"segment_id": "segment-1"}]))

    def fail_create_dependency(_config):
        raise AssertionError(
            "indexing dependencies should not load for invalid video filename"
        )

    monkeypatch.setattr(indexing_step, "create_embedding_model", fail_create_dependency)
    monkeypatch.setattr(indexing_step, "create_vector_collection", fail_create_dependency)

    with pytest.raises(ValueError, match="video_filename"):
        run_indexing(
            str(enriched_segments_path),
            video_filename,
            {"database": {"collection_name": "unused"}},
        )


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ({"segment_id": "segment-1"}, "JSON array"),
        (["not a segment"], "index 0"),
        ([{"summary": "missing id"}], "segment_id"),
        ([{"segment_id": " "}], "segment_id"),
        ([{"segment_id": "segment-1", "full_transcript": ["bad"]}], "full_transcript"),
        ([{"segment_id": "segment-1", "summary": {"bad": "data"}}], "summary"),
        ([{"segment_id": "segment-1", "speakers": "Alice"}], "speakers"),
        ([{"segment_id": "segment-1", "keywords": [42]}], "keywords"),
        (
            [{"segment_id": "segment-1", "consolidated_visual_captions": "caption"}],
            "consolidated_visual_captions",
        ),
        (
            [{"segment_id": "segment-1", "consolidated_actions": [None]}],
            "consolidated_actions",
        ),
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
        raise AssertionError(
            "indexing dependencies should not load for invalid segment data"
        )

    monkeypatch.setattr(indexing_step, "create_embedding_model", fail_create_dependency)
    monkeypatch.setattr(indexing_step, "create_vector_collection", fail_create_dependency)

    with pytest.raises(ValueError, match=message):
        run_indexing(
            str(enriched_segments_path),
            "demo-video",
            {"database": {"collection_name": "unused"}},
        )


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ([{"segment_id": "segment-1", "end_time": 1}], "start_time"),
        ([{"segment_id": "segment-1", "start_time": 0}], "end_time"),
        ([{"segment_id": "segment-1", "start_time": "bad"}], "start_time"),
        ([{"segment_id": "segment-1", "start_time": True}], "start_time"),
        ([{"segment_id": "segment-1", "start_time": -1}], "start_time"),
        ([{"segment_id": "segment-1", "start_time": float("nan")}], "start_time"),
        ([{"segment_id": "segment-1", "start_time": 0, "end_time": None}], "end_time"),
        ([{"segment_id": "segment-1", "start_time": 0, "end_time": float("inf")}], "end_time"),
        ([{"segment_id": "segment-1", "start_time": 4, "end_time": 3}], "end_time"),
    ],
)
def test_run_indexing_rejects_invalid_timing_metadata_before_creating_dependencies(
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
