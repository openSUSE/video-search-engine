import json

import pytest

from ingestion_pipeline.jobs import IngestionJob, decode_job_message, encode_job_message


def test_ingestion_job_message_round_trip_omits_empty_fields():
    job = IngestionJob(video_path="/data/videos/demo.mp4", output_dir=None, title=None, year=2024)

    message = json.loads(encode_job_message(job))
    decoded = decode_job_message(json.dumps(message).encode("utf-8"))

    assert message == {"video_path": "/data/videos/demo.mp4", "year": 2024}
    assert decoded == job


def test_ingestion_job_rejects_missing_video_path():
    with pytest.raises(ValueError, match="video_path"):
        IngestionJob(video_path="")


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"video_path": 123}, "video_path"),
        ({"video_path": "  "}, "video_path"),
        ({"video_path": "/data/videos/demo.mp4", "output_dir": 42}, "output_dir"),
        ({"video_path": "/data/videos/demo.mp4", "title": 42}, "title"),
        ({"video_path": "/data/videos/demo.mp4", "year": "2024"}, "year"),
        ({"video_path": "/data/videos/demo.mp4", "year": True}, "year"),
    ],
)
def test_decode_job_message_rejects_invalid_field_types(payload, error):
    with pytest.raises(ValueError, match=error):
        decode_job_message(json.dumps(payload).encode("utf-8"))


def test_ingestion_job_builds_pipeline_kwargs_with_default_output_dir():
    job = IngestionJob(video_path="/data/videos/demo.mp4", title="Demo", year=2024)

    assert job.to_pipeline_kwargs("data/processed") == {
        "video_path": "/data/videos/demo.mp4",
        "output_dir": "data/processed",
        "title": "Demo",
        "year": 2024,
    }
