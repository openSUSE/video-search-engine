import json

import pytest

from app.ui.speaker_support import (
    load_transcript_segments,
    load_speaker_map,
    normalize_speaker_map,
    processed_video_folders,
    reset_speaker_session_for_video,
    resolve_video_path,
    save_speaker_map_if_complete,
    speaker_artifact_paths,
    speaker_ids_from_transcript,
    validate_speaker_ids_from_transcript,
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


def test_normalize_speaker_map_trims_ids_and_names():
    assert normalize_speaker_map({" SPEAKER_00 ": "  Alice  "}) == {
        "SPEAKER_00": "Alice",
    }


def test_normalize_speaker_map_rejects_invalid_entries():
    assert normalize_speaker_map({"SPEAKER_00": "   "}) is None
    assert normalize_speaker_map({"SPEAKER_00": {"name": "Alice"}}) is None
    assert normalize_speaker_map({"SPEAKER_00": "Alice", " SPEAKER_00 ": "Alicia"}) is None


def test_normalize_speaker_map_allows_empty_map():
    assert normalize_speaker_map({}) == {}


def test_load_speaker_map_normalizes_existing_map(tmp_path):
    speaker_map_path = tmp_path / "speaker_map.json"
    speaker_map_path.write_text('{" SPEAKER_00 ": "  Alice  "}')

    assert load_speaker_map(speaker_map_path) == {"SPEAKER_00": "Alice"}


def test_load_speaker_map_returns_empty_map_for_invalid_existing_map(tmp_path):
    speaker_map_path = tmp_path / "speaker_map.json"
    speaker_map_path.write_text('{"SPEAKER_00": "   "}')

    assert load_speaker_map(speaker_map_path) == {}


def test_load_transcript_segments_validates_display_fields(tmp_path):
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        '[{"start": 1, "text": " hello ", "speaker": " SPEAKER_00 "}]'
    )

    assert load_transcript_segments(transcript_path) == [
        {"start": 1.0, "text": " hello ", "speaker": "SPEAKER_00"}
    ]


@pytest.mark.parametrize(
    ("transcript", "message"),
    [
        ({"speaker": "SPEAKER_00"}, "JSON array"),
        (["not a segment"], "index 0"),
        ([{"text": "hello"}], "start"),
        ([{"start": True, "text": "hello"}], "start"),
        ([{"start": -1, "text": "hello"}], "start"),
        ([{"start": 0}], "text"),
        ([{"start": 0, "text": "hello", "speaker": 7}], "speaker"),
    ],
)
def test_load_transcript_segments_rejects_malformed_display_data(
    tmp_path,
    transcript,
    message,
):
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps(transcript))

    with pytest.raises(ValueError, match=message):
        load_transcript_segments(transcript_path)


def test_save_speaker_map_if_complete_writes_empty_map_for_no_transcript_speakers(tmp_path):
    speaker_map_path = tmp_path / "speaker_map.json"

    saved = save_speaker_map_if_complete(speaker_map_path, {}, [])

    assert saved is True
    assert speaker_map_path.read_text() == "{}"


def test_save_speaker_map_if_complete_rejects_incomplete_maps(tmp_path):
    speaker_map_path = tmp_path / "speaker_map.json"

    saved = save_speaker_map_if_complete(
        speaker_map_path,
        {"SPEAKER_00": "Alice"},
        ["SPEAKER_00", "SPEAKER_01"],
    )

    assert saved is False
    assert not speaker_map_path.exists()


def test_speaker_ids_from_transcript_returns_sorted_non_empty_strings():
    transcript = [
        {"speaker": " SPEAKER_01 "},
        {"speaker": "SPEAKER_00"},
        {"speaker": "SPEAKER_00"},
        {"speaker": ""},
        {"text": "missing speaker"},
        "not a segment",
    ]

    assert speaker_ids_from_transcript(transcript) == ["SPEAKER_00", "SPEAKER_01"]


def test_validate_speaker_ids_from_transcript_returns_sorted_non_empty_strings():
    transcript = [
        {"speaker": " SPEAKER_01 "},
        {"speaker": "SPEAKER_00"},
        {"speaker": ""},
        {"text": "missing speaker"},
    ]

    assert validate_speaker_ids_from_transcript(transcript) == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]


@pytest.mark.parametrize(
    ("transcript", "message"),
    [
        ({"speaker": "SPEAKER_00"}, "JSON array"),
        (["not a segment"], "index 0"),
        ([{"speaker": 7}], "speaker"),
    ],
)
def test_validate_speaker_ids_from_transcript_rejects_malformed_segments(
    transcript,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_speaker_ids_from_transcript(transcript)
