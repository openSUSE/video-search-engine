import json
import sys
import types
from pathlib import Path

import pytest

from ingestion_pipeline.steps import step_01_extraction as extraction_step
from ingestion_pipeline.steps.step_01_extraction import (
    _get_paths,
    align_transcript_to_shots,
    create_final_analysis_file,
    detect_audio_events_per_shot,
    detect_shot_boundaries,
    detect_actions_per_shot,
    generate_visual_captions,
    run_extraction,
)


def _extraction_config():
    return {
        "filenames": {
            "audio": "audio.mp3",
            "raw_transcript": "custom_raw_transcript.json",
            "transcript": "aligned.json",
            "shots": "shots.json",
            "audio_events": "audio_events.json",
            "visual_details": "visual_details.json",
            "actions": "actions.json",
            "final_analysis": "final_analysis.json",
        }
    }


def test_get_paths_uses_configured_raw_transcript_name(tmp_path):
    paths = _get_paths(str(tmp_path), _extraction_config())

    assert paths["transcript_raw"] == str(tmp_path / "custom_raw_transcript.json")


@pytest.mark.parametrize(
    ("opened", "fps", "error_match"),
    [
        (False, 30.0, "Cannot open video file"),
        (True, 0.0, "Could not read a valid FPS"),
    ],
)
def test_detect_shot_boundaries_rejects_invalid_video_before_model_load(
    monkeypatch,
    tmp_path,
    opened,
    fps,
    error_match,
):
    release_calls = []

    class FakeVideoCapture:
        def __init__(self, path):
            self.path = path

        def isOpened(self):
            return opened

        def get(self, _property):
            return fps

        def release(self):
            release_calls.append(self.path)

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=FakeVideoCapture,
        CAP_PROP_FPS=5,
    )
    fake_transnet = types.ModuleType("transnetv2_pytorch")

    class FailingTransNetV2:
        def __init__(self):
            raise AssertionError("TransNet should not load for invalid video metadata")

    fake_transnet.TransNetV2 = FailingTransNetV2
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "transnetv2_pytorch", fake_transnet)

    with pytest.raises(IOError, match=error_match):
        detect_shot_boundaries("bad-video.mp4", str(tmp_path / "shots.json"))

    assert release_calls == ["bad-video.mp4"]


def test_generate_visual_captions_rejects_unreadable_video_before_model_load(
    monkeypatch,
    tmp_path,
):
    release_calls = []

    class FakeVideoCapture:
        def __init__(self, path):
            self.path = path

        def isOpened(self):
            return False

        def release(self):
            release_calls.append(self.path)

    fake_cv2 = types.SimpleNamespace(VideoCapture=FakeVideoCapture)
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = object()
    fake_transformers = types.ModuleType("transformers")

    class FailingBlipProcessor:
        @classmethod
        def from_pretrained(cls, _name):
            raise AssertionError("BLIP should not load for unreadable videos")

    fake_transformers.BlipProcessor = FailingBlipProcessor
    fake_transformers.BlipForConditionalGeneration = FailingBlipProcessor
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(IOError, match="Cannot open video file"):
        generate_visual_captions(
            "bad-video.mp4",
            [{"shot_id": "shot_0001", "start_frame": 0, "end_frame": 10}],
            str(tmp_path / "visual_details.json"),
            {
                "general": {"device": "cpu"},
                "models": {"visual_captioning": {"name": "blip"}},
                "parameters": {"visual_captioning": {"max_new_tokens": 50}},
            },
        )

    assert release_calls == ["bad-video.mp4"]


def test_generate_visual_captions_keeps_shot_coverage_when_frame_read_fails(
    monkeypatch,
    tmp_path,
):
    output_path = tmp_path / "visual_details.json"
    calls = {"released": False}

    class FakeVideoCapture:
        def __init__(self, path):
            self.path = path

        def isOpened(self):
            return True

        def set(self, _property, _value):
            pass

        def read(self):
            return False, None

        def release(self):
            calls["released"] = True

    class FakeLoader:
        @classmethod
        def from_pretrained(cls, _name):
            return cls()

        def to(self, _device):
            return self

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=FakeVideoCapture,
        CAP_PROP_POS_FRAMES=1,
    )
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = object()
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.BlipProcessor = FakeLoader
    fake_transformers.BlipForConditionalGeneration = FakeLoader
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    generate_visual_captions(
        "demo.mp4",
        [{"shot_id": "shot_0001", "start_frame": 0, "end_frame": 10}],
        str(output_path),
        {
            "general": {"device": "cpu"},
            "models": {"visual_captioning": {"name": "blip"}},
            "parameters": {"visual_captioning": {"max_new_tokens": 50}},
        },
    )

    assert json.loads(output_path.read_text()) == [
        {"shot_id": "shot_0001", "caption": ""}
    ]
    assert calls["released"] is True


def test_detect_actions_rejects_unreadable_video_before_model_load(
    monkeypatch,
    tmp_path,
):
    release_calls = []

    class FakeVideoCapture:
        def __init__(self, path):
            self.path = path

        def isOpened(self):
            return False

        def release(self):
            release_calls.append(self.path)

    fake_cv2 = types.SimpleNamespace(VideoCapture=FakeVideoCapture)
    fake_transformers = types.ModuleType("transformers")

    class FailingVideoMAEProcessor:
        @classmethod
        def from_pretrained(cls, _name):
            raise AssertionError("VideoMAE should not load for unreadable videos")

    fake_transformers.VideoMAEImageProcessor = FailingVideoMAEProcessor
    fake_transformers.VideoMAEForVideoClassification = FailingVideoMAEProcessor
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(IOError, match="Cannot open video file"):
        detect_actions_per_shot(
            "bad-video.mp4",
            [{"shot_id": "shot_0001", "start_frame": 0, "end_frame": 20}],
            str(tmp_path / "actions.json"),
            {
                "general": {"device": "cpu"},
                "models": {"action_recognition": {"name": "videomae"}},
                "parameters": {"action_recognition": {"num_frames": 16, "top_n": 3}},
            },
        )

    assert release_calls == ["bad-video.mp4"]


def test_per_shot_extractors_skip_empty_scenes_without_model_load(
    monkeypatch,
    tmp_path,
):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("per-shot extractors should not load media or models for no scenes")

    fake_librosa = types.ModuleType("librosa")
    fake_librosa.load = fail_if_called

    fake_cv2 = types.SimpleNamespace(VideoCapture=fail_if_called)

    fake_transformers = types.ModuleType("transformers")

    class FailingModelLoader:
        @classmethod
        def from_pretrained(cls, _name):
            fail_if_called()

    fake_transformers.AutoProcessor = FailingModelLoader
    fake_transformers.AutoModelForAudioClassification = FailingModelLoader
    fake_transformers.BlipProcessor = FailingModelLoader
    fake_transformers.BlipForConditionalGeneration = FailingModelLoader
    fake_transformers.VideoMAEImageProcessor = FailingModelLoader
    fake_transformers.VideoMAEForVideoClassification = FailingModelLoader

    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))

    config = {
        "general": {"device": "cpu"},
        "models": {
            "audio_events": {"name": "ast"},
            "visual_captioning": {"name": "blip"},
            "action_recognition": {"name": "videomae"},
        },
        "parameters": {
            "audio": {"sample_rate": 16000},
            "audio_events": {"top_n": 3, "confidence_threshold": 0.1},
            "visual_captioning": {"max_new_tokens": 50},
            "action_recognition": {"num_frames": 16, "top_n": 3},
        },
    }

    audio_events_path = tmp_path / "audio_events.json"
    visual_details_path = tmp_path / "visual_details.json"
    actions_path = tmp_path / "actions.json"

    detect_audio_events_per_shot("missing.mp3", [], str(audio_events_path), config)
    generate_visual_captions("missing.mp4", [], str(visual_details_path), config)
    detect_actions_per_shot("missing.mp4", [], str(actions_path), config)

    assert json.loads(audio_events_path.read_text()) == []
    assert json.loads(visual_details_path.read_text()) == []
    assert json.loads(actions_path.read_text()) == []


def test_detect_audio_events_rejects_unreadable_audio_before_model_load(
    monkeypatch,
    tmp_path,
):
    fake_librosa = types.ModuleType("librosa")

    def failing_load(*_args, **_kwargs):
        raise OSError("bad audio")

    fake_librosa.load = failing_load
    fake_transformers = types.ModuleType("transformers")

    class FailingAstLoader:
        @classmethod
        def from_pretrained(cls, _name):
            raise AssertionError("AST should not load for unreadable audio")

    fake_transformers.AutoProcessor = FailingAstLoader
    fake_transformers.AutoModelForAudioClassification = FailingAstLoader
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    with pytest.raises(OSError, match="bad audio"):
        detect_audio_events_per_shot(
            "bad-audio.mp3",
            [{"shot_id": "shot_0001", "start_time_sec": 0.0, "end_time_sec": 1.0}],
            str(tmp_path / "audio_events.json"),
            {
                "general": {"device": "cpu"},
                "models": {"audio_events": {"name": "ast"}},
                "parameters": {
                    "audio": {"sample_rate": 16000},
                    "audio_events": {"top_n": 3, "confidence_threshold": 0.1},
                },
            },
        )


def test_run_extraction_uses_injected_config_and_metadata_fetcher(tmp_path):
    config = _extraction_config()
    output_dir = tmp_path / "processed"
    video_dir = output_dir / "demo"
    video_dir.mkdir(parents=True)
    paths = _get_paths(str(video_dir), config)

    Path(paths["shots"]).write_text("[]")
    Path(paths["audio"]).write_text("audio")
    Path(paths["transcript_raw"]).write_text("[]")
    Path(paths["transcript_aligned"]).write_text("[]")
    Path(paths["audio_events"]).write_text("[]")
    Path(paths["visual_details"]).write_text("[]")
    Path(paths["actions"]).write_text("[]")
    Path(paths["final_analysis"]).write_text("[]")
    calls = []

    def fake_metadata_fetcher(title, year):
        calls.append({"title": title, "year": year})
        return {"title": "Fetched Demo", "synopsis": "Fetched synopsis."}

    result = run_extraction(
        video_path=str(tmp_path / "demo.mp4"),
        base_output_dir=str(output_dir),
        video_title="Demo",
        video_year=2024,
        config=config,
        metadata_fetcher=fake_metadata_fetcher,
    )

    assert result == paths["final_analysis"]
    assert calls == [{"title": "Demo", "year": 2024}]
    metadata = json.loads((video_dir / "video_metadata.json").read_text())
    assert metadata["title"] == "Fetched Demo"
    assert metadata["synopsis"] == "Fetched synopsis."
    assert "logline" not in metadata


def test_run_extraction_ignores_blank_fetched_metadata_fields(tmp_path):
    config = _extraction_config()
    output_dir = tmp_path / "processed"
    video_dir = output_dir / "demo"
    video_dir.mkdir(parents=True)
    paths = _get_paths(str(video_dir), config)

    for path in paths.values():
        Path(path).write_text("[]")

    def fake_metadata_fetcher(_title, _year):
        return {
            "title": "   ",
            "synopsis": "",
            "genre": None,
        }

    run_extraction(
        video_path=str(tmp_path / "demo.mp4"),
        base_output_dir=str(output_dir),
        video_title="Demo",
        config=config,
        metadata_fetcher=fake_metadata_fetcher,
    )

    metadata = json.loads((video_dir / "video_metadata.json").read_text())
    assert metadata == {
        "title": "Demo",
        "synopsis": "No synopsis provided.",
        "genre": "N/A",
    }


def test_run_extraction_strips_fetched_metadata_fields(tmp_path):
    config = _extraction_config()
    output_dir = tmp_path / "processed"
    video_dir = output_dir / "demo"
    video_dir.mkdir(parents=True)
    paths = _get_paths(str(video_dir), config)

    for path in paths.values():
        Path(path).write_text("[]")

    def fake_metadata_fetcher(_title, _year):
        return {
            "title": "  Fetched Demo  ",
            "synopsis": "  Fetched synopsis.  ",
            "genre": "  Drama  ",
        }

    run_extraction(
        video_path=str(tmp_path / "demo.mp4"),
        base_output_dir=str(output_dir),
        video_title="Demo",
        config=config,
        metadata_fetcher=fake_metadata_fetcher,
    )

    metadata = json.loads((video_dir / "video_metadata.json").read_text())
    assert metadata == {
        "title": "Fetched Demo",
        "synopsis": "Fetched synopsis.",
        "genre": "Drama",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"video_path": "   ", "base_output_dir": "processed"}, "video_path"),
        ({"video_path": "demo.mp4", "base_output_dir": ""}, "base_output_dir"),
        ({"video_path": None, "base_output_dir": "processed"}, "video_path"),
        ({"video_path": "demo.mp4", "base_output_dir": None}, "base_output_dir"),
    ],
)
def test_run_extraction_rejects_invalid_required_paths_before_loading_config(
    monkeypatch,
    kwargs,
    message,
):
    def fail_load_config():
        raise AssertionError("config should not load for invalid required paths")

    monkeypatch.setattr(extraction_step, "_load_config", fail_load_config)

    with pytest.raises(ValueError, match=message):
        run_extraction(**kwargs)


@pytest.mark.parametrize("video_title", [123, True])
def test_run_extraction_rejects_invalid_title_before_loading_config(
    monkeypatch,
    tmp_path,
    video_title,
):
    def fail_load_config():
        raise AssertionError("config should not load for an invalid title")

    monkeypatch.setattr(extraction_step, "_load_config", fail_load_config)

    with pytest.raises(ValueError, match="title"):
        run_extraction(
            video_path=str(tmp_path / "demo.mp4"),
            base_output_dir=str(tmp_path / "processed"),
            video_title=video_title,
        )


def test_run_extraction_treats_blank_title_as_missing_before_metadata_lookup(tmp_path):
    config = _extraction_config()
    output_dir = tmp_path / "processed"
    video_dir = output_dir / "demo"
    video_dir.mkdir(parents=True)
    paths = _get_paths(str(video_dir), config)

    for path in paths.values():
        Path(path).write_text("[]")

    def fail_metadata_fetcher(_title, _year):
        raise AssertionError("blank titles should not trigger metadata lookup")

    run_extraction(
        video_path=str(tmp_path / "demo.mp4"),
        base_output_dir=str(output_dir),
        video_title="   ",
        config=config,
        metadata_fetcher=fail_metadata_fetcher,
    )

    metadata = json.loads((video_dir / "video_metadata.json").read_text())
    assert metadata["title"] == "demo"
    assert metadata["synopsis"] == "No synopsis provided."


@pytest.mark.parametrize("video_year", [0, -1, True])
def test_run_extraction_rejects_invalid_year_before_metadata_lookup(
    tmp_path,
    video_year,
):
    calls = []

    def fake_metadata_fetcher(title, year):
        calls.append({"title": title, "year": year})
        return {"title": "Fetched Demo"}

    with pytest.raises(ValueError, match="year"):
        run_extraction(
            video_path=str(tmp_path / "demo.mp4"),
            base_output_dir=str(tmp_path / "processed"),
            video_title="Demo",
            video_year=video_year,
            config=_extraction_config(),
            metadata_fetcher=fake_metadata_fetcher,
        )

    assert calls == []


def test_run_extraction_rejects_invalid_year_before_loading_config(monkeypatch, tmp_path):
    def fail_load_config():
        raise AssertionError("config should not load for an invalid year")

    monkeypatch.setattr(extraction_step, "_load_config", fail_load_config)

    with pytest.raises(ValueError, match="year"):
        run_extraction(
            video_path=str(tmp_path / "demo.mp4"),
            base_output_dir=str(tmp_path / "processed"),
            video_title="Demo",
            video_year=0,
        )


def test_run_extraction_preserves_existing_metadata_when_no_title_is_provided(tmp_path):
    config = _extraction_config()
    output_dir = tmp_path / "processed"
    video_dir = output_dir / "demo"
    video_dir.mkdir(parents=True)
    paths = _get_paths(str(video_dir), config)

    for path in paths.values():
        Path(path).write_text("[]")

    existing_metadata = {
        "title": "Fetched Demo",
        "synopsis": "Fetched synopsis.",
        "genre": "Drama",
    }
    metadata_path = video_dir / "video_metadata.json"
    metadata_path.write_text(json.dumps(existing_metadata))

    run_extraction(
        video_path=str(tmp_path / "demo.mp4"),
        base_output_dir=str(output_dir),
        config=config,
    )

    assert json.loads(metadata_path.read_text()) == existing_metadata


def test_run_extraction_refreshes_existing_final_analysis_from_current_artifacts(tmp_path):
    config = _extraction_config()
    output_dir = tmp_path / "processed"
    video_dir = output_dir / "demo"
    video_dir.mkdir(parents=True)
    paths = _get_paths(str(video_dir), config)

    Path(paths["shots"]).write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "shot_index": 1,
                    "start_frame": 0,
                    "end_frame": 24,
                    "start_time_sec": 0.0,
                    "end_time_sec": 1.0,
                }
            ]
        )
    )
    Path(paths["audio"]).write_text("audio")
    Path(paths["transcript_raw"]).write_text("[]")
    Path(paths["transcript_aligned"]).write_text(
        json.dumps(
            [
                {
                    "start": 0.0,
                    "end": 0.8,
                    "text": "current line",
                    "speaker": "Alice",
                    "shot_id": "shot_0001",
                }
            ]
        )
    )
    Path(paths["visual_details"]).write_text(
        json.dumps([{"shot_id": "shot_0001", "caption": "current caption"}])
    )
    Path(paths["audio_events"]).write_text(
        json.dumps([{"shot_id": "shot_0001", "events": [{"event": "speech"}]}])
    )
    Path(paths["actions"]).write_text(
        json.dumps([{"shot_id": "shot_0001", "actions": [{"action": "standing"}]}])
    )
    Path(paths["final_analysis"]).write_text(
        json.dumps([{"shot_id": "stale_shot", "visual_caption": "old caption"}])
    )

    run_extraction(
        video_path=str(tmp_path / "demo.mp4"),
        base_output_dir=str(output_dir),
        config=config,
    )

    assert json.loads(Path(paths["final_analysis"]).read_text()) == [
        {
            "shot_id": "shot_0001",
            "shot_index": 1,
            "time_start_sec": 0.0,
            "time_end_sec": 1.0,
            "frame_start": 0,
            "frame_end": 24,
            "visual_caption": "current caption",
            "detected_actions": [{"action": "standing"}],
            "audio_events": [{"event": "speech"}],
            "transcript_segments": [
                {
                    "start": 0.0,
                    "end": 0.8,
                    "text": "current line",
                    "speaker": "Alice",
                }
            ],
        }
    ]


def test_run_extraction_rejects_invalid_cached_shots_before_downstream_work(
    monkeypatch,
    tmp_path,
):
    config = _extraction_config()
    output_dir = tmp_path / "processed"
    video_dir = output_dir / "demo"
    video_dir.mkdir(parents=True)
    paths = _get_paths(str(video_dir), config)
    Path(paths["shots"]).write_text(json.dumps({"shot_id": "shot_0001"}))

    def fail_extract_audio(*_args, **_kwargs):
        raise AssertionError("downstream extraction should not run for invalid cached shots")

    monkeypatch.setattr(extraction_step, "extract_audio", fail_extract_audio)

    with pytest.raises(ValueError, match="shot boundaries"):
        run_extraction(
            video_path=str(tmp_path / "demo.mp4"),
            base_output_dir=str(output_dir),
            config=config,
        )


def test_run_extraction_rejects_duplicate_cached_shot_ids_before_downstream_work(
    monkeypatch,
    tmp_path,
):
    config = _extraction_config()
    output_dir = tmp_path / "processed"
    video_dir = output_dir / "demo"
    video_dir.mkdir(parents=True)
    paths = _get_paths(str(video_dir), config)
    Path(paths["shots"]).write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "shot_index": 1,
                    "start_frame": 0,
                    "end_frame": 10,
                    "start_time_sec": 0.0,
                    "end_time_sec": 1.0,
                },
                {
                    "shot_id": " shot_0001 ",
                    "shot_index": 2,
                    "start_frame": 11,
                    "end_frame": 20,
                    "start_time_sec": 1.0,
                    "end_time_sec": 2.0,
                },
            ]
        )
    )

    def fail_extract_audio(*_args, **_kwargs):
        raise AssertionError(
            "downstream extraction should not run for duplicate shot IDs"
        )

    monkeypatch.setattr(extraction_step, "extract_audio", fail_extract_audio)

    with pytest.raises(ValueError, match="duplicate shot_id"):
        run_extraction(
            video_path=str(tmp_path / "demo.mp4"),
            base_output_dir=str(output_dir),
            config=config,
        )


def test_run_extraction_rejects_overlapping_cached_shots_before_downstream_work(
    monkeypatch,
    tmp_path,
):
    config = _extraction_config()
    output_dir = tmp_path / "processed"
    video_dir = output_dir / "demo"
    video_dir.mkdir(parents=True)
    paths = _get_paths(str(video_dir), config)
    Path(paths["shots"]).write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "shot_index": 1,
                    "start_frame": 0,
                    "end_frame": 20,
                    "start_time_sec": 0.0,
                    "end_time_sec": 2.0,
                },
                {
                    "shot_id": "shot_0002",
                    "shot_index": 2,
                    "start_frame": 10,
                    "end_frame": 30,
                    "start_time_sec": 1.0,
                    "end_time_sec": 3.0,
                },
            ]
        )
    )

    def fail_extract_audio(*_args, **_kwargs):
        raise AssertionError(
            "downstream extraction should not run for overlapping cached shots"
        )

    monkeypatch.setattr(extraction_step, "extract_audio", fail_extract_audio)

    with pytest.raises(ValueError, match="overlap"):
        run_extraction(
            video_path=str(tmp_path / "demo.mp4"),
            base_output_dir=str(output_dir),
            config=config,
        )


@pytest.mark.parametrize(
    ("shot_updates", "message"),
    [
        ({"shot_index": 0}, "shot_index"),
        ({"start_frame": -1}, "start_frame"),
        ({"start_frame": 10, "end_frame": 9}, "end_frame"),
        ({"start_time_sec": -0.1}, "start_time_sec"),
        ({"start_time_sec": float("nan")}, "start_time_sec"),
        ({"start_time_sec": 2.0, "end_time_sec": 1.0}, "end_time_sec"),
        ({"end_time_sec": float("inf")}, "end_time_sec"),
    ],
)
def test_run_extraction_rejects_inconsistent_cached_shot_boundaries(
    monkeypatch,
    tmp_path,
    shot_updates,
    message,
):
    config = _extraction_config()
    output_dir = tmp_path / "processed"
    video_dir = output_dir / "demo"
    video_dir.mkdir(parents=True)
    paths = _get_paths(str(video_dir), config)
    shot = {
        "shot_id": "shot_0001",
        "shot_index": 1,
        "start_frame": 0,
        "end_frame": 10,
        "start_time_sec": 0.0,
        "end_time_sec": 1.0,
    }
    shot.update(shot_updates)
    Path(paths["shots"]).write_text(json.dumps([shot]))

    def fail_extract_audio(*_args, **_kwargs):
        raise AssertionError("downstream extraction should not run for invalid cached shots")

    monkeypatch.setattr(extraction_step, "extract_audio", fail_extract_audio)

    with pytest.raises(ValueError, match=message):
        run_extraction(
            video_path=str(tmp_path / "demo.mp4"),
            base_output_dir=str(output_dir),
            config=config,
        )


def test_align_transcript_to_shots_writes_valid_aligned_segments(tmp_path):
    raw_transcript_path = tmp_path / "transcript_raw.json"
    aligned_transcript_path = tmp_path / "aligned.json"
    raw_transcript_path.write_text(
        json.dumps(
            [
                {
                    "start": 0.25,
                    "end": 0.75,
                    "text": "hello",
                    "speaker": "SPEAKER_00",
                }
            ]
        )
    )

    align_transcript_to_shots(
        str(raw_transcript_path),
        [
            {
                "shot_id": "shot_0001",
                "start_time_sec": 0.0,
                "end_time_sec": 1.0,
            }
        ],
        str(aligned_transcript_path),
    )

    assert json.loads(aligned_transcript_path.read_text()) == [
        {
            "start": 0.25,
            "end": 0.75,
            "text": "hello",
            "speaker": "SPEAKER_00",
            "shot_id": "shot_0001",
        }
    ]


def test_align_transcript_to_shots_assigns_final_end_boundary_to_last_shot(tmp_path):
    raw_transcript_path = tmp_path / "transcript_raw.json"
    aligned_transcript_path = tmp_path / "aligned.json"
    raw_transcript_path.write_text(
        json.dumps(
            [
                {
                    "start": 1.0,
                    "end": 1.0,
                    "text": "closing word",
                    "speaker": "SPEAKER_00",
                }
            ]
        )
    )

    align_transcript_to_shots(
        str(raw_transcript_path),
        [
            {
                "shot_id": "shot_0001",
                "start_time_sec": 0.0,
                "end_time_sec": 1.0,
            }
        ],
        str(aligned_transcript_path),
    )

    [aligned_segment] = json.loads(aligned_transcript_path.read_text())
    assert aligned_segment["shot_id"] == "shot_0001"


@pytest.mark.parametrize(
    ("raw_transcript", "message"),
    [
        ({"start": 0.0, "end": 1.0}, "JSON array"),
        (["not a segment"], "index 0"),
        ([{"end": 1.0, "text": "hello"}], "start"),
        ([{"start": True, "end": 1.0, "text": "hello"}], "start"),
        ([{"start": -0.1, "end": 1.0, "text": "hello"}], "start"),
        ([{"start": float("nan"), "end": 1.0, "text": "hello"}], "start"),
        ([{"start": 2.0, "end": 1.0, "text": "hello"}], "end"),
        ([{"start": 0.0, "end": float("inf"), "text": "hello"}], "end"),
        ([{"start": 0.0, "end": 1.0}], "text"),
        ([{"start": 0.0, "end": 1.0, "text": "hello", "speaker": 7}], "speaker"),
    ],
)
def test_align_transcript_to_shots_rejects_invalid_raw_transcript_before_writing(
    tmp_path,
    raw_transcript,
    message,
):
    raw_transcript_path = tmp_path / "transcript_raw.json"
    aligned_transcript_path = tmp_path / "aligned.json"
    raw_transcript_path.write_text(json.dumps(raw_transcript))

    with pytest.raises(ValueError, match=message):
        align_transcript_to_shots(
            str(raw_transcript_path),
            [
                {
                    "shot_id": "shot_0001",
                    "start_time_sec": 0.0,
                    "end_time_sec": 1.0,
                }
            ],
            str(aligned_transcript_path),
        )

    assert not aligned_transcript_path.exists()


def test_create_final_analysis_file_combines_intermediate_outputs(tmp_path):
    paths = {
        "shots": str(tmp_path / "shots.json"),
        "visual_details": str(tmp_path / "visual_details.json"),
        "audio_events": str(tmp_path / "audio_events.json"),
        "transcript_aligned": str(tmp_path / "aligned.json"),
        "actions": str(tmp_path / "actions.json"),
        "final_analysis": str(tmp_path / "final_analysis.json"),
    }
    Path(paths["shots"]).write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "shot_index": 1,
                    "start_time_sec": 0.0,
                    "end_time_sec": 2.0,
                    "start_frame": 0,
                    "end_frame": 48,
                }
            ]
        )
    )
    Path(paths["visual_details"]).write_text(
        json.dumps([{"shot_id": "shot_0001", "caption": "a station platform"}])
    )
    Path(paths["audio_events"]).write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "events": [{"event": " speech ", "score": 0.9}],
                }
            ]
        )
    )
    Path(paths["transcript_aligned"]).write_text(
        json.dumps(
            [
                {
                    "start": 0.1,
                    "end": 1.2,
                    "text": "hello",
                    "speaker": "Alice",
                    "shot_id": "shot_0001",
                }
            ]
        )
    )
    Path(paths["actions"]).write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "actions": [{"action": " standing ", "score": 0.8}],
                }
            ]
        )
    )

    create_final_analysis_file(paths)

    assert json.loads(Path(paths["final_analysis"]).read_text()) == [
        {
            "shot_id": "shot_0001",
            "shot_index": 1,
            "time_start_sec": 0.0,
            "time_end_sec": 2.0,
            "frame_start": 0,
            "frame_end": 48,
            "visual_caption": "a station platform",
            "detected_actions": [{"action": "standing", "score": 0.8}],
            "audio_events": [{"event": "speech", "score": 0.9}],
            "transcript_segments": [
                {
                    "start": 0.1,
                    "end": 1.2,
                    "text": "hello",
                    "speaker": "Alice",
                }
            ],
        }
    ]


@pytest.mark.parametrize(
    ("artifact_name", "invalid_data", "message"),
    [
        ("visual_details", [{"shot_id": "shot_0001"}], "visual details"),
        (
            "audio_events",
            [{"shot_id": "shot_0001", "events": "speech"}],
            "audio events",
        ),
        (
            "audio_events",
            [{"shot_id": "shot_0001", "events": [{"event": "   "}]}],
            "audio events",
        ),
        (
            "audio_events",
            [
                {
                    "shot_id": "shot_0001",
                    "events": [{"event": "speech", "score": "0.9"}],
                }
            ],
            "audio events",
        ),
        (
            "audio_events",
            [
                {
                    "shot_id": "shot_0001",
                    "events": [{"event": "speech", "score": True}],
                }
            ],
            "audio events",
        ),
        (
            "actions",
            [{"shot_id": "shot_0001", "actions": [{"score": 0.8}]}],
            "actions",
        ),
        (
            "actions",
            [{"shot_id": "shot_0001", "actions": [{"action": ""}]}],
            "actions",
        ),
        (
            "actions",
            [
                {
                    "shot_id": "shot_0001",
                    "actions": [{"action": "standing", "score": -0.1}],
                }
            ],
            "actions",
        ),
        (
            "actions",
            [
                {
                    "shot_id": "shot_0001",
                    "actions": [{"action": "standing", "score": float("nan")}],
                }
            ],
            "actions",
        ),
        (
            "transcript_aligned",
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "speaker": "Alice",
                    "shot_id": 7,
                }
            ],
            "aligned transcript",
        ),
        (
            "transcript_aligned",
            [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "Alice"}],
            "aligned transcript",
        ),
    ],
)
def test_create_final_analysis_file_rejects_invalid_cached_artifacts_before_writing(
    tmp_path,
    artifact_name,
    invalid_data,
    message,
):
    paths = {
        "shots": str(tmp_path / "shots.json"),
        "visual_details": str(tmp_path / "visual_details.json"),
        "audio_events": str(tmp_path / "audio_events.json"),
        "transcript_aligned": str(tmp_path / "aligned.json"),
        "actions": str(tmp_path / "actions.json"),
        "final_analysis": str(tmp_path / "final_analysis.json"),
    }
    valid_artifacts = {
        "shots": [
            {
                "shot_id": "shot_0001",
                "shot_index": 1,
                "start_time_sec": 0.0,
                "end_time_sec": 2.0,
                "start_frame": 0,
                "end_frame": 48,
            }
        ],
        "visual_details": [{"shot_id": "shot_0001", "caption": "a station platform"}],
        "audio_events": [{"shot_id": "shot_0001", "events": [{"event": "speech"}]}],
        "transcript_aligned": [
            {
                "start": 0.1,
                "end": 1.2,
                "text": "hello",
                "speaker": "Alice",
                "shot_id": "shot_0001",
            }
        ],
        "actions": [{"shot_id": "shot_0001", "actions": [{"action": "standing"}]}],
    }
    valid_artifacts[artifact_name] = invalid_data
    for name, data in valid_artifacts.items():
        Path(paths[name]).write_text(json.dumps(data))

    with pytest.raises(ValueError, match=message):
        create_final_analysis_file(paths)

    assert not Path(paths["final_analysis"]).exists()


def test_create_final_analysis_file_rejects_artifacts_for_unknown_shots(tmp_path):
    paths = {
        "shots": str(tmp_path / "shots.json"),
        "visual_details": str(tmp_path / "visual_details.json"),
        "audio_events": str(tmp_path / "audio_events.json"),
        "transcript_aligned": str(tmp_path / "aligned.json"),
        "actions": str(tmp_path / "actions.json"),
        "final_analysis": str(tmp_path / "final_analysis.json"),
    }
    Path(paths["shots"]).write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "shot_index": 1,
                    "start_time_sec": 0.0,
                    "end_time_sec": 2.0,
                    "start_frame": 0,
                    "end_frame": 48,
                }
            ]
        )
    )
    Path(paths["visual_details"]).write_text(
        json.dumps([{"shot_id": "shot_9999", "caption": "stale caption"}])
    )
    Path(paths["audio_events"]).write_text(json.dumps([]))
    Path(paths["transcript_aligned"]).write_text(json.dumps([]))
    Path(paths["actions"]).write_text(json.dumps([]))

    with pytest.raises(ValueError, match="unknown shot_id"):
        create_final_analysis_file(paths)

    assert not Path(paths["final_analysis"]).exists()


@pytest.mark.parametrize(
    ("artifact_name", "missing_artifact_data", "message"),
    [
        (
            "visual_details",
            [{"shot_id": "shot_0001", "caption": "a station platform"}],
            "visual details is missing shot_id: shot_0002",
        ),
        (
            "audio_events",
            [{"shot_id": "shot_0001", "events": [{"event": "speech"}]}],
            "audio events is missing shot_id: shot_0002",
        ),
        (
            "actions",
            [{"shot_id": "shot_0001", "actions": [{"action": "standing"}]}],
            "actions is missing shot_id: shot_0002",
        ),
    ],
)
def test_create_final_analysis_file_rejects_per_shot_artifacts_missing_known_shots(
    tmp_path,
    artifact_name,
    missing_artifact_data,
    message,
):
    paths = {
        "shots": str(tmp_path / "shots.json"),
        "visual_details": str(tmp_path / "visual_details.json"),
        "audio_events": str(tmp_path / "audio_events.json"),
        "transcript_aligned": str(tmp_path / "aligned.json"),
        "actions": str(tmp_path / "actions.json"),
        "final_analysis": str(tmp_path / "final_analysis.json"),
    }
    valid_artifacts = {
        "shots": [
            {
                "shot_id": "shot_0001",
                "shot_index": 1,
                "start_time_sec": 0.0,
                "end_time_sec": 2.0,
                "start_frame": 0,
                "end_frame": 48,
            },
            {
                "shot_id": "shot_0002",
                "shot_index": 2,
                "start_time_sec": 2.0,
                "end_time_sec": 4.0,
                "start_frame": 49,
                "end_frame": 96,
            },
        ],
        "visual_details": [
            {"shot_id": "shot_0001", "caption": "a station platform"},
            {"shot_id": "shot_0002", "caption": "a train"},
        ],
        "audio_events": [
            {"shot_id": "shot_0001", "events": [{"event": "speech"}]},
            {"shot_id": "shot_0002", "events": []},
        ],
        "transcript_aligned": [],
        "actions": [
            {"shot_id": "shot_0001", "actions": [{"action": "standing"}]},
            {"shot_id": "shot_0002", "actions": []},
        ],
    }
    valid_artifacts[artifact_name] = missing_artifact_data
    for name, data in valid_artifacts.items():
        Path(paths[name]).write_text(json.dumps(data))

    with pytest.raises(ValueError, match=message):
        create_final_analysis_file(paths)

    assert not Path(paths["final_analysis"]).exists()
