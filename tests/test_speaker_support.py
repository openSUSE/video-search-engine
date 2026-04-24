from app.ui.speaker_support import (
    processed_video_folders,
    reset_speaker_session_for_video,
    resolve_video_path,
    speaker_artifact_paths,
)


def test_processed_video_folders_returns_sorted_directories(tmp_path):
    (tmp_path / "zeta").mkdir()
    (tmp_path / "alpha").mkdir()
    (tmp_path / "notes.txt").write_text("ignore me")

    assert processed_video_folders(tmp_path) == ["alpha", "zeta"]


def test_speaker_artifact_paths_use_configured_filenames_and_video_extension(tmp_path):
    processed_dir = tmp_path / "processed"
    video_dir = tmp_path / "videos"
    (processed_dir / "demo").mkdir(parents=True)
    video_dir.mkdir()
    (video_dir / "demo.mov").write_bytes(b"video")

    paths = speaker_artifact_paths(
        processed_dir,
        "demo",
        video_dir,
        {
            "filenames": {
                "transcript": "custom_transcript.json",
                "speaker_map": "custom_speaker_map.json",
            }
        },
    )

    assert paths.video_dir == processed_dir / "demo"
    assert paths.transcript == processed_dir / "demo" / "custom_transcript.json"
    assert paths.video == video_dir / "demo.mov"
    assert paths.speaker_map == processed_dir / "demo" / "custom_speaker_map.json"


def test_resolve_video_path_defaults_to_mp4_when_no_candidate_exists(tmp_path):
    assert resolve_video_path(tmp_path, "missing") == tmp_path / "missing.mp4"


def test_reset_speaker_session_for_video_clears_previous_video_state():
    state = {
        "selected_video_folder": "old",
        "speaker_map": {"SPEAKER_00": "Alice"},
        "video_start_time": 42,
        "current_transcript_data": [{"text": "old"}],
    }

    reset_speaker_session_for_video(state, "new")

    assert state == {
        "selected_video_key": "new",
        "selected_video_folder": "new",
        "speaker_map": {},
        "video_start_time": 0,
        "current_transcript_data": None,
    }


def test_reset_speaker_session_for_video_includes_base_dir_in_selection():
    state = {
        "selected_video_key": "processed-a::demo",
        "selected_video_folder": "demo",
        "speaker_map": {"SPEAKER_00": "Alice"},
        "video_start_time": 42,
        "current_transcript_data": [{"text": "old"}],
    }

    reset_speaker_session_for_video(state, "demo", "processed-b")

    assert state["selected_video_key"] == "processed-b::demo"
    assert state["speaker_map"] == {}
