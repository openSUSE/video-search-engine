import json
from pathlib import Path

from app import main as speaker_app


def test_speaker_ui_template_does_not_render_transcript_data_with_inner_html():
    template = Path("app/ui/index.html").read_text()

    assert "innerHTML" not in template


def test_save_speaker_map_writes_configured_output(monkeypatch, tmp_path):
    video_path = tmp_path / "videos" / "demo.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"")

    output_dir = tmp_path / "processed"
    monkeypatch.setattr(speaker_app, "VIDEO_PATH", str(video_path))
    monkeypatch.setattr(speaker_app, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(
        speaker_app,
        "CONFIG",
        {
            "filenames": {"speaker_map": "speaker_map.json"},
            "ui": {"host": "127.0.0.1", "port": 5050},
        },
    )
    monkeypatch.setattr(speaker_app, "_request_shutdown_async", lambda: None)

    response = speaker_app.app.test_client().post(
        "/api/save_map",
        json={"speaker_map": {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}},
    )

    speaker_map_path = output_dir / "demo" / "speaker_map.json"
    assert response.status_code == 200
    assert json.loads(speaker_map_path.read_text()) == {
        "SPEAKER_00": "Alice",
        "SPEAKER_01": "Bob",
    }


def test_save_speaker_map_rejects_invalid_payload():
    response = speaker_app.app.test_client().post("/api/save_map", json=["not", "an", "object"])

    assert response.status_code == 400
