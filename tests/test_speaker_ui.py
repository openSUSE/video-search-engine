import json
from pathlib import Path

from app import main as speaker_app


def test_speaker_ui_template_does_not_render_transcript_data_with_inner_html():
    template = Path("app/ui/index.html").read_text()

    assert "innerHTML" not in template


def test_speaker_ui_template_handles_transcript_load_errors():
    template = Path("app/ui/index.html").read_text()

    assert "response.ok" in template
    assert "Array.isArray(data.transcript)" in template
    assert "saveButton.disabled = true" in template


def test_speaker_flask_ui_url_uses_loopback_for_wildcard_host(monkeypatch):
    monkeypatch.setattr(
        speaker_app,
        "CONFIG",
        {
            "filenames": {"speaker_map": "speaker_map.json"},
            "ui": {"host": "0.0.0.0", "port": 5050},
        },
    )

    assert speaker_app._ui_url() == "http://127.0.0.1:5050"


def test_speaker_flask_ui_url_uses_runtime_port_override(monkeypatch):
    monkeypatch.setattr(
        speaker_app,
        "CONFIG",
        {
            "filenames": {"speaker_map": "speaker_map.json"},
            "ui": {"host": "0.0.0.0", "port": 5050},
        },
    )
    monkeypatch.setattr(speaker_app, "SERVER_HOST", "0.0.0.0", raising=False)
    monkeypatch.setattr(speaker_app, "SERVER_PORT", 6060, raising=False)

    assert speaker_app._ui_url() == "http://127.0.0.1:6060"


def test_get_data_returns_validated_transcript(monkeypatch, tmp_path):
    video_path = tmp_path / "videos" / "demo.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"")
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            [
                {
                    "start": 1,
                    "end": 2,
                    "text": "hello",
                    "speaker": " SPEAKER_00 ",
                }
            ]
        )
    )

    monkeypatch.setattr(speaker_app, "VIDEO_PATH", str(video_path))
    monkeypatch.setattr(speaker_app, "TRANSCRIPT_PATH", str(transcript_path))

    response = speaker_app.app.test_client().get("/api/data")

    assert response.status_code == 200
    assert response.get_json() == {
        "transcript": [
            {
                "start": 1.0,
                "end": 2.0,
                "text": "hello",
                "speaker": "SPEAKER_00",
            }
        ],
        "video_filename": "demo.mp4",
    }


def test_get_data_rejects_malformed_transcript(monkeypatch, tmp_path):
    video_path = tmp_path / "videos" / "demo.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"")
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps({"text": "not a segment array"}))

    monkeypatch.setattr(speaker_app, "VIDEO_PATH", str(video_path))
    monkeypatch.setattr(speaker_app, "TRANSCRIPT_PATH", str(transcript_path))

    response = speaker_app.app.test_client().get("/api/data")

    assert response.status_code == 500
    assert response.get_json() == {"error": "Could not read transcript file."}


def test_get_data_returns_not_found_for_missing_transcript(monkeypatch, tmp_path):
    video_path = tmp_path / "videos" / "demo.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"")

    monkeypatch.setattr(speaker_app, "VIDEO_PATH", str(video_path))
    monkeypatch.setattr(speaker_app, "TRANSCRIPT_PATH", str(tmp_path / "missing.json"))

    response = speaker_app.app.test_client().get("/api/data")

    assert response.status_code == 404
    assert response.get_json() == {"error": "Transcript file not found."}


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


def test_save_speaker_map_trims_names_before_writing(monkeypatch, tmp_path):
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
        json={"speaker_map": {"SPEAKER_00": "  Alice  "}},
    )

    speaker_map_path = output_dir / "demo" / "speaker_map.json"
    assert response.status_code == 200
    assert json.loads(speaker_map_path.read_text()) == {"SPEAKER_00": "Alice"}


def test_save_speaker_map_rejects_invalid_payload():
    response = speaker_app.app.test_client().post("/api/save_map", json=["not", "an", "object"])

    assert response.status_code == 400


def test_save_speaker_map_rejects_invalid_speaker_names():
    client = speaker_app.app.test_client()

    for speaker_map in [
        {"SPEAKER_00": ""},
        {"SPEAKER_00": "   "},
        {"SPEAKER_00": {"name": "Alice"}},
    ]:
        response = client.post("/api/save_map", json={"speaker_map": speaker_map})

        assert response.status_code == 400


def test_save_speaker_map_rejects_missing_transcript_speakers(monkeypatch, tmp_path):
    video_path = tmp_path / "videos" / "demo.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"")
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            [
                {"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "hello"},
                {"start": 1, "end": 2, "speaker": "SPEAKER_01", "text": "reply"},
            ]
        )
    )

    output_dir = tmp_path / "processed"
    monkeypatch.setattr(speaker_app, "VIDEO_PATH", str(video_path))
    monkeypatch.setattr(speaker_app, "TRANSCRIPT_PATH", str(transcript_path))
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
        json={"speaker_map": {"SPEAKER_00": "Alice"}},
    )

    assert response.status_code == 400
    assert not (output_dir / "demo" / "speaker_map.json").exists()


def test_save_speaker_map_rejects_malformed_transcript(monkeypatch, tmp_path):
    video_path = tmp_path / "videos" / "demo.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"")
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps([{"speaker": "SPEAKER_00", "text": "hello"}]))

    output_dir = tmp_path / "processed"
    monkeypatch.setattr(speaker_app, "VIDEO_PATH", str(video_path))
    monkeypatch.setattr(speaker_app, "TRANSCRIPT_PATH", str(transcript_path))
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
        json={"speaker_map": {"SPEAKER_00": "Alice"}},
    )

    assert response.status_code == 500
    assert not (output_dir / "demo" / "speaker_map.json").exists()


def test_save_speaker_map_allows_empty_map_when_transcript_has_no_speakers(monkeypatch, tmp_path):
    video_path = tmp_path / "videos" / "demo.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"")
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps([{"start": 0, "end": 1, "text": "music"}]))

    output_dir = tmp_path / "processed"
    monkeypatch.setattr(speaker_app, "VIDEO_PATH", str(video_path))
    monkeypatch.setattr(speaker_app, "TRANSCRIPT_PATH", str(transcript_path))
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
        json={"speaker_map": {}},
    )

    speaker_map_path = output_dir / "demo" / "speaker_map.json"
    assert response.status_code == 200
    assert json.loads(speaker_map_path.read_text()) == {}
