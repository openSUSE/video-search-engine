import json
import sys
import types

import pytest

from ingestion_pipeline.steps import step_02_segmentation as segmentation_step
from ingestion_pipeline.steps.step_02_segmentation import create_embedding_model, run_segmentation


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": texts, "kwargs": kwargs})
        vectors = {
            "hello there": [1.0, 0.0],
            "still hello": [1.0, 0.0],
            "new topic": [0.0, 1.0],
            "quiet room": [1.0, 0.0],
            "fast car": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]


def _cached_segment(**updates):
    segment = {
        "segment_id": "segment_0001",
        "segment_index": 1,
        "start_time": 0.0,
        "end_time": 1.0,
        "duration_sec": 1.0,
        "speakers": ["Alice"],
        "full_transcript": "hello",
        "consolidated_visual_captions": ["room"],
        "consolidated_audio_events": [],
        "consolidated_actions": [],
        "shot_count": 1,
        "shot_ids": ["shot_0001"],
    }
    segment.update(updates)
    return segment


def test_create_embedding_model_uses_configured_name_and_device(monkeypatch):
    calls = {}
    fake_module = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name, device):
            calls["model_name"] = model_name
            calls["device"] = device

    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    create_embedding_model(
        {
            "general": {"device": "cuda"},
            "models": {"embedding": {"name": "custom-embedding-model"}},
        }
    )

    assert calls == {
        "model_name": "custom-embedding-model",
        "device": "cuda",
    }


def test_run_segmentation_uses_injected_model_and_configured_output_name(tmp_path):
    analysis_path = tmp_path / "analysis.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    analysis_path.write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "text": "hello there",
                            "speaker": "SPEAKER_00",
                        }
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                },
                {
                    "shot_id": "shot_0002",
                    "time_start_sec": 1.0,
                    "time_end_sec": 2.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [
                        {
                            "start": 1.0,
                            "end": 2.0,
                            "text": "still hello",
                            "speaker": "SPEAKER_00",
                        }
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                },
                {
                    "shot_id": "shot_0003",
                    "time_start_sec": 2.0,
                    "time_end_sec": 3.0,
                    "visual_caption": "fast car",
                    "transcript_segments": [
                        {
                            "start": 2.0,
                            "end": 3.0,
                            "text": "new topic",
                            "speaker": "SPEAKER_01",
                        }
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                },
            ]
        )
    )
    speaker_map_path.write_text(json.dumps({"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}))
    embedding_model = FakeEmbeddingModel()

    output_path = run_segmentation(
        video_path="unused.mp4",
        analysis_path=str(analysis_path),
        speaker_map_path=str(speaker_map_path),
        config={"filenames": {"final_segments": "custom_segments.json"}},
        embedding_model=embedding_model,
    )

    assert output_path == str(tmp_path / "custom_segments.json")
    segments = json.loads((tmp_path / "custom_segments.json").read_text())
    assert [segment["shot_ids"] for segment in segments] == [
        ["shot_0001", "shot_0002"],
        ["shot_0003"],
    ]
    assert segments[0]["full_transcript"] == "hello there still hello"
    assert segments[0]["speakers"] == ["Alice"]
    assert segments[1]["speakers"] == ["Bob"]
    assert embedding_model.calls == [
        {
            "texts": ["hello there", "still hello", "new topic"],
            "kwargs": {"show_progress_bar": True, "normalize_embeddings": True},
        },
        {
            "texts": ["quiet room", "quiet room", "fast car"],
            "kwargs": {"show_progress_bar": True, "normalize_embeddings": True},
        },
    ]


def test_run_segmentation_preserves_empty_speakers_for_transcripts_without_ids(tmp_path):
    analysis_path = tmp_path / "analysis.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    analysis_path.write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [
                        {
                            "text": "ambient music",
                        },
                        {
                            "text": "wind rises",
                            "speaker": "  ",
                        },
                    ],
                    "audio_events": [{"event": " music "}],
                    "detected_actions": [{"action": " walking "}],
                }
            ]
        )
    )
    speaker_map_path.write_text("{}")

    class AnyTextEmbeddingModel:
        def encode(self, texts, **_kwargs):
            return [[1.0, 0.0] for _text in texts]

    output_path = run_segmentation(
        video_path="unused.mp4",
        analysis_path=str(analysis_path),
        speaker_map_path=str(speaker_map_path),
        config={"filenames": {"final_segments": "segments.json"}},
        embedding_model=AnyTextEmbeddingModel(),
    )

    segments = json.loads((tmp_path / "segments.json").read_text())
    assert output_path == str(tmp_path / "segments.json")
    assert segments[0]["speakers"] == []
    assert segments[0]["full_transcript"] == "ambient music wind rises"
    assert segments[0]["consolidated_audio_events"] == ["music"]
    assert segments[0]["consolidated_actions"] == ["walking"]


def test_run_segmentation_normalizes_visual_captions_before_embedding_and_output(tmp_path):
    analysis_path = tmp_path / "analysis.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    analysis_path.write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "  quiet room  ",
                    "transcript_segments": [{"text": "ambient"}],
                    "audio_events": [],
                    "detected_actions": [],
                },
                {
                    "shot_id": "shot_0002",
                    "time_start_sec": 1.0,
                    "time_end_sec": 2.0,
                    "visual_caption": "   ",
                    "transcript_segments": [{"text": "ambient"}],
                    "audio_events": [],
                    "detected_actions": [],
                },
            ]
        )
    )
    speaker_map_path.write_text("{}")

    class StableEmbeddingModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, **kwargs):
            self.calls.append({"texts": texts, "kwargs": kwargs})
            return [[1.0, 0.0] for _text in texts]

    embedding_model = StableEmbeddingModel()

    output_path = run_segmentation(
        video_path="unused.mp4",
        analysis_path=str(analysis_path),
        speaker_map_path=str(speaker_map_path),
        config={"filenames": {"final_segments": "segments.json"}},
        embedding_model=embedding_model,
    )

    segments = json.loads((tmp_path / "segments.json").read_text())
    assert output_path == str(tmp_path / "segments.json")
    assert embedding_model.calls[1]["texts"] == ["quiet room", ""]
    assert segments[0]["consolidated_visual_captions"] == ["quiet room"]


def test_run_segmentation_skips_when_cached_output_is_valid(tmp_path):
    output_path = tmp_path / "custom_segments.json"
    output_path.write_text(json.dumps([_cached_segment()]))

    assert run_segmentation(
        video_path="unused.mp4",
        analysis_path=str(tmp_path / "missing_analysis.json"),
        speaker_map_path=str(tmp_path / "missing_speaker_map.json"),
        config={"filenames": {"final_segments": output_path.name}},
    ) == str(output_path)


def test_run_segmentation_skips_cached_output_when_current_analysis_matches(
    monkeypatch,
    tmp_path,
):
    analysis_path = tmp_path / "analysis.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    output_path = tmp_path / "custom_segments.json"
    analysis_path.write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "room",
                    "transcript_segments": [
                        {"text": "hello", "speaker": "SPEAKER_00"}
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                }
            ]
        )
    )
    speaker_map_path.write_text(json.dumps({"SPEAKER_00": "Alice"}))
    output_path.write_text(json.dumps([_cached_segment()]))

    def fail_create_embedding_model(_config):
        raise AssertionError("matching cached segmentation should skip model loading")

    monkeypatch.setattr(
        segmentation_step,
        "create_embedding_model",
        fail_create_embedding_model,
    )

    assert run_segmentation(
        video_path="unused.mp4",
        analysis_path=str(analysis_path),
        speaker_map_path=str(speaker_map_path),
        config={"filenames": {"final_segments": output_path.name}},
    ) == str(output_path)


def test_run_segmentation_recomputes_cached_output_when_analysis_content_changed(
    tmp_path,
):
    analysis_path = tmp_path / "analysis.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    output_path = tmp_path / "custom_segments.json"
    analysis_path.write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "current room",
                    "transcript_segments": [
                        {"text": "current line", "speaker": "SPEAKER_00"}
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                }
            ]
        )
    )
    speaker_map_path.write_text(json.dumps({"SPEAKER_00": "Alice"}))
    output_path.write_text(
        json.dumps(
            [
                _cached_segment(
                    full_transcript="old line",
                    consolidated_visual_captions=["old room"],
                )
            ]
        )
    )

    class AnyTextEmbeddingModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, **kwargs):
            self.calls.append({"texts": texts, "kwargs": kwargs})
            return [[1.0, 0.0] for _text in texts]

    embedding_model = AnyTextEmbeddingModel()

    assert run_segmentation(
        video_path="unused.mp4",
        analysis_path=str(analysis_path),
        speaker_map_path=str(speaker_map_path),
        config={"filenames": {"final_segments": output_path.name}},
        embedding_model=embedding_model,
    ) == str(output_path)

    [segment] = json.loads(output_path.read_text())
    assert segment["full_transcript"] == "current line"
    assert segment["consolidated_visual_captions"] == ["current room"]
    assert embedding_model.calls


@pytest.mark.parametrize(
    ("cached_segments", "message"),
    [
        ({"segment_id": "segment_0001"}, "JSON array"),
        (["not a segment"], "index 0"),
        ([_cached_segment(segment_id=" ")], "segment_id"),
        ([_cached_segment(segment_id="custom-segment")], "segment_id"),
        ([_cached_segment(segment_index=2)], "segment_index"),
        ([_cached_segment(start_time=-1)], "start_time"),
        ([_cached_segment(start_time=float("nan"))], "start_time"),
        ([_cached_segment(start_time=2, end_time=1)], "end_time"),
        ([_cached_segment(start_time=0, end_time=2, duration_sec=99)], "duration_sec"),
        ([_cached_segment(end_time=float("inf"))], "end_time"),
        ([_cached_segment(speakers="Alice")], "speakers"),
        ([_cached_segment(shot_ids=[7])], "shot_ids"),
    ],
)
def test_run_segmentation_rejects_malformed_cached_output_before_skipping(
    tmp_path,
    cached_segments,
    message,
):
    output_path = tmp_path / "custom_segments.json"
    output_path.write_text(json.dumps(cached_segments))

    with pytest.raises(ValueError, match=message):
        run_segmentation(
            video_path="unused.mp4",
            analysis_path=str(tmp_path / "missing_analysis.json"),
            speaker_map_path=str(tmp_path / "missing_speaker_map.json"),
            config={"filenames": {"final_segments": output_path.name}},
        )


def test_run_segmentation_rejects_duplicate_cached_segment_ids_before_skipping(
    tmp_path,
):
    output_path = tmp_path / "custom_segments.json"
    output_path.write_text(
        json.dumps(
            [
                _cached_segment(segment_id="segment_0001"),
                _cached_segment(
                    segment_id=" segment_0001 ",
                    segment_index=2,
                    start_time=2.0,
                    end_time=3.0,
                    shot_ids=["shot_0002"],
                ),
            ]
        )
    )

    with pytest.raises(ValueError, match="duplicate segment_id"):
        run_segmentation(
            video_path="unused.mp4",
            analysis_path=str(tmp_path / "missing_analysis.json"),
            speaker_map_path=str(tmp_path / "missing_speaker_map.json"),
            config={"filenames": {"final_segments": output_path.name}},
        )


def test_run_segmentation_rejects_overlapping_cached_segments_before_skipping(
    tmp_path,
):
    output_path = tmp_path / "custom_segments.json"
    output_path.write_text(
        json.dumps(
            [
                _cached_segment(
                    segment_id="segment_0001",
                    start_time=0.0,
                    end_time=5.0,
                    duration_sec=5.0,
                ),
                _cached_segment(
                    segment_id="segment_0002",
                    segment_index=2,
                    start_time=4.0,
                    end_time=6.0,
                    duration_sec=2.0,
                    shot_ids=["shot_0002"],
                ),
            ]
        )
    )

    with pytest.raises(ValueError, match="overlaps previous segment"):
        run_segmentation(
            video_path="unused.mp4",
            analysis_path=str(tmp_path / "missing_analysis.json"),
            speaker_map_path=str(tmp_path / "missing_speaker_map.json"),
            config={"filenames": {"final_segments": output_path.name}},
        )


@pytest.mark.parametrize(
    ("bad_call_index", "message"),
    [
        (0, "dialogue embedding count"),
        (1, "visual embedding count"),
    ],
)
def test_run_segmentation_rejects_embedding_count_mismatch_before_writing_output(
    tmp_path,
    bad_call_index,
    message,
):
    analysis_path = tmp_path / "analysis.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    output_path = tmp_path / "segments.json"
    analysis_path.write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [
                        {
                            "text": "hello there",
                            "speaker": "SPEAKER_00",
                        }
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                },
                {
                    "shot_id": "shot_0002",
                    "time_start_sec": 1.0,
                    "time_end_sec": 2.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [
                        {
                            "text": "still hello",
                            "speaker": "SPEAKER_00",
                        }
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                },
            ]
        )
    )
    speaker_map_path.write_text(json.dumps({"SPEAKER_00": "Alice"}))

    class ShortEmbeddingModel:
        def __init__(self):
            self.call_count = 0

        def encode(self, texts, **kwargs):
            call_index = self.call_count
            self.call_count += 1
            if call_index == bad_call_index:
                return [[1.0, 0.0]]
            return [[1.0, 0.0] for _text in texts]

    with pytest.raises(ValueError, match=message):
        run_segmentation(
            video_path="unused.mp4",
            analysis_path=str(analysis_path),
            speaker_map_path=str(speaker_map_path),
            config={"filenames": {"final_segments": output_path.name}},
            embedding_model=ShortEmbeddingModel(),
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("bad_call_index", "message"),
    [
        (0, "dialogue embeddings must have consistent dimensions"),
        (1, "visual embeddings must have consistent dimensions"),
    ],
)
def test_run_segmentation_rejects_embedding_dimension_mismatch_before_scoring(
    tmp_path,
    bad_call_index,
    message,
):
    analysis_path = tmp_path / "analysis.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    output_path = tmp_path / "segments.json"
    analysis_path.write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [
                        {
                            "text": "hello there",
                            "speaker": "SPEAKER_00",
                        }
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                },
                {
                    "shot_id": "shot_0002",
                    "time_start_sec": 1.0,
                    "time_end_sec": 2.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [
                        {
                            "text": "still hello",
                            "speaker": "SPEAKER_00",
                        }
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                },
            ]
        )
    )
    speaker_map_path.write_text(json.dumps({"SPEAKER_00": "Alice"}))

    class MismatchedEmbeddingModel:
        def __init__(self):
            self.call_count = 0

        def encode(self, texts, **kwargs):
            call_index = self.call_count
            self.call_count += 1
            if call_index == bad_call_index:
                return [[1.0, 0.0], [1.0, 0.0, 0.0]]
            return [[1.0, 0.0] for _text in texts]

    with pytest.raises(ValueError, match=message):
        run_segmentation(
            video_path="unused.mp4",
            analysis_path=str(analysis_path),
            speaker_map_path=str(speaker_map_path),
            config={"filenames": {"final_segments": output_path.name}},
            embedding_model=MismatchedEmbeddingModel(),
        )

    assert not output_path.exists()


def test_run_segmentation_rejects_nonfinite_embeddings_before_scoring(tmp_path):
    analysis_path = tmp_path / "analysis.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    output_path = tmp_path / "segments.json"
    analysis_path.write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [
                        {
                            "text": "hello there",
                            "speaker": "SPEAKER_00",
                        }
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                }
            ]
        )
    )
    speaker_map_path.write_text(json.dumps({"SPEAKER_00": "Alice"}))

    class NonfiniteEmbeddingModel:
        def encode(self, texts, **kwargs):
            return [[float("nan")] for _text in texts]

    with pytest.raises(ValueError, match="dialogue embeddings must be numeric vectors"):
        run_segmentation(
            video_path="unused.mp4",
            analysis_path=str(analysis_path),
            speaker_map_path=str(speaker_map_path),
            config={"filenames": {"final_segments": output_path.name}},
            embedding_model=NonfiniteEmbeddingModel(),
        )

    assert not output_path.exists()


@pytest.mark.parametrize("encoded_vector", ["12", b"12", bytearray(b"12")])
def test_run_segmentation_rejects_string_like_embedding_vectors_before_scoring(
    tmp_path,
    encoded_vector,
):
    analysis_path = tmp_path / "analysis.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    output_path = tmp_path / "segments.json"
    analysis_path.write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [
                        {
                            "text": "hello there",
                            "speaker": "SPEAKER_00",
                        }
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                }
            ]
        )
    )
    speaker_map_path.write_text(json.dumps({"SPEAKER_00": "Alice"}))

    class StringLikeVectorEmbeddingModel:
        def encode(self, texts, **kwargs):
            return [encoded_vector for _text in texts]

    with pytest.raises(ValueError, match="dialogue embeddings must be numeric vectors"):
        run_segmentation(
            video_path="unused.mp4",
            analysis_path=str(analysis_path),
            speaker_map_path=str(speaker_map_path),
            config={"filenames": {"final_segments": output_path.name}},
            embedding_model=StringLikeVectorEmbeddingModel(),
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("analysis_data", "message"),
    [
        ({"shot_id": "shot_0001"}, "JSON array"),
        (["not a shot"], "shot at index 0"),
        (
            [
                {
                    "shot_id": "shot_0001",
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [],
                }
            ],
            "time_start_sec",
        ),
        (
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": -0.1,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [],
                }
            ],
            "time_start_sec",
        ),
        (
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": float("nan"),
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [],
                }
            ],
            "time_start_sec",
        ),
        (
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 2.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [],
                }
            ],
            "time_end_sec",
        ),
        (
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": float("inf"),
                    "visual_caption": "quiet room",
                    "transcript_segments": [],
                }
            ],
            "time_end_sec",
        ),
        (
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 5.0,
                    "time_end_sec": 6.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [],
                },
                {
                    "shot_id": "shot_0002",
                    "time_start_sec": 4.0,
                    "time_end_sec": 5.0,
                    "visual_caption": "same room",
                    "transcript_segments": [],
                },
            ],
            "overlaps previous shot",
        ),
        (
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": "hello",
                }
            ],
            "transcript_segments",
        ),
        (
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [{"speaker": "SPEAKER_00"}],
                }
            ],
            "text",
        ),
        (
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [],
                    "audio_events": [{"event": "   "}],
                    "detected_actions": [],
                }
            ],
            "audio_events",
        ),
        (
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [],
                    "audio_events": [],
                    "detected_actions": [{"action": ""}],
                }
            ],
            "detected_actions",
        ),
    ],
)
def test_run_segmentation_rejects_invalid_analysis_before_embedding_setup(
    monkeypatch,
    tmp_path,
    analysis_data,
    message,
):
    analysis_path = tmp_path / "analysis.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    analysis_path.write_text(json.dumps(analysis_data))
    speaker_map_path.write_text(json.dumps({"SPEAKER_00": "Alice"}))

    def fail_create_embedding_model(_config):
        raise AssertionError("embedding model should not load for invalid analysis data")

    monkeypatch.setattr(
        segmentation_step,
        "create_embedding_model",
        fail_create_embedding_model,
    )

    with pytest.raises(ValueError, match=message):
        run_segmentation(
            video_path="unused.mp4",
            analysis_path=str(analysis_path),
            speaker_map_path=str(speaker_map_path),
            config={"filenames": {"final_segments": "segments.json"}},
        )


def test_run_segmentation_rejects_duplicate_shot_ids_before_embedding_setup(
    monkeypatch,
    tmp_path,
):
    analysis_path = tmp_path / "analysis.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    analysis_path.write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [],
                    "audio_events": [],
                    "detected_actions": [],
                },
                {
                    "shot_id": " shot_0001 ",
                    "time_start_sec": 1.0,
                    "time_end_sec": 2.0,
                    "visual_caption": "same room",
                    "transcript_segments": [],
                    "audio_events": [],
                    "detected_actions": [],
                },
            ]
        )
    )
    speaker_map_path.write_text("{}")

    def fail_create_embedding_model(_config):
        raise AssertionError("embedding model should not load for duplicate shot IDs")

    monkeypatch.setattr(segmentation_step, "create_embedding_model", fail_create_embedding_model)

    with pytest.raises(ValueError, match="duplicate shot_id"):
        run_segmentation(
            video_path="unused.mp4",
            analysis_path=str(analysis_path),
            speaker_map_path=str(speaker_map_path),
            config={"filenames": {"final_segments": "segments.json"}},
        )


@pytest.mark.parametrize(
    ("speaker_map", "message"),
    [
        (["not a map"], "speaker map"),
        ({"SPEAKER_00": " "}, "speaker map"),
        ({"SPEAKER_00": "Alice", " SPEAKER_00 ": "Alicia"}, "speaker map"),
    ],
)
def test_run_segmentation_rejects_invalid_speaker_map_before_embedding_setup(
    monkeypatch,
    tmp_path,
    speaker_map,
    message,
):
    analysis_path = tmp_path / "analysis.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    analysis_path.write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [
                        {
                            "text": "hello there",
                            "speaker": "SPEAKER_00",
                        }
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                }
            ]
        )
    )
    speaker_map_path.write_text(json.dumps(speaker_map))

    def fail_create_embedding_model(_config):
        raise AssertionError("embedding model should not load for invalid speaker map")

    monkeypatch.setattr(segmentation_step, "create_embedding_model", fail_create_embedding_model)

    with pytest.raises(ValueError, match=message):
        run_segmentation(
            video_path="unused.mp4",
            analysis_path=str(analysis_path),
            speaker_map_path=str(speaker_map_path),
            config={"filenames": {"final_segments": "segments.json"}},
        )


def test_run_segmentation_rejects_incomplete_speaker_map_before_embedding_setup(
    monkeypatch,
    tmp_path,
):
    analysis_path = tmp_path / "analysis.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    analysis_path.write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [
                        {
                            "text": "hello there",
                            "speaker": "SPEAKER_00",
                        },
                        {
                            "text": "reply",
                            "speaker": "SPEAKER_01",
                        },
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                }
            ]
        )
    )
    speaker_map_path.write_text(json.dumps({"SPEAKER_00": "Alice"}))

    def fail_create_embedding_model(_config):
        raise AssertionError("embedding model should not load for incomplete speaker maps")

    monkeypatch.setattr(segmentation_step, "create_embedding_model", fail_create_embedding_model)

    with pytest.raises(ValueError, match="SPEAKER_01"):
        run_segmentation(
            video_path="unused.mp4",
            analysis_path=str(analysis_path),
            speaker_map_path=str(speaker_map_path),
            config={"filenames": {"final_segments": "segments.json"}},
        )
