import json
import sys
import types
from pathlib import Path

import pytest

from ingestion_pipeline.steps.step_01_extraction import (
    _get_paths,
    create_final_analysis_file,
    detect_shot_boundaries,
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

    run_extraction(
        video_path=str(tmp_path / "demo.mp4"),
        base_output_dir=str(output_dir),
        video_title="Demo",
        video_year=2024,
        config=config,
        metadata_fetcher=fake_metadata_fetcher,
    )

    assert calls == [{"title": "Demo", "year": 2024}]
    metadata = json.loads((video_dir / "video_metadata.json").read_text())
    assert metadata["title"] == "Fetched Demo"
    assert metadata["synopsis"] == "Fetched synopsis."
    assert "logline" not in metadata


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
        json.dumps([{"shot_id": "shot_0001", "events": [{"event": "speech", "score": 0.9}]}])
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
        json.dumps([{"shot_id": "shot_0001", "actions": [{"action": "standing", "score": 0.8}]}])
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
