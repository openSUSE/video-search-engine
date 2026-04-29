from app.ui.search_state import (
    ensure_search_session_state,
    reset_search_session_for_video,
)


def test_ensure_search_session_state_sets_defaults_without_clobbering_existing_values():
    state = {
        "video_path": "data/videos/demo.mp4",
        "start_time": 42,
        "search_results": [{"id": "segment_0001"}],
        "last_search_query": "find this",
    }

    ensure_search_session_state(state)

    assert state["video_path"] == "data/videos/demo.mp4"
    assert state["start_time"] == 42
    assert state["search_results"] == [{"id": "segment_0001"}]
    assert state["last_search_query"] == "find this"


def test_ensure_search_session_state_initializes_last_search_query_to_none():
    state = {}

    ensure_search_session_state(state)

    assert state["last_search_query"] is None


def test_reset_search_session_for_video_preserves_state_for_same_selection():
    state = {
        "selected_video_file": "demo.mp4",
        "video_path": "data/videos/demo.mp4",
        "video_filename_clean": "demo",
        "start_time": 42,
        "search_results": [{"id": "segment_0001"}],
    }

    reset_search_session_for_video(
        state,
        selected_video_file="demo.mp4",
        video_path="data/videos/demo.mp4",
        video_filename_clean="demo",
    )

    assert state["start_time"] == 42
    assert state["search_results"] == [{"id": "segment_0001"}]


def test_reset_search_session_for_video_clears_stale_results_for_new_selection():
    state = {
        "selected_video_file": "old.mp4",
        "video_path": "data/videos/old.mp4",
        "video_filename_clean": "old",
        "start_time": 42,
        "search_results": [{"id": "old::segment_0001"}],
        "last_search_query": "stale query",
    }

    reset_search_session_for_video(
        state,
        selected_video_file="new.mov",
        video_path="data/videos/new.mov",
        video_filename_clean="new",
    )

    assert state == {
        "selected_video_file": "new.mov",
        "video_path": "data/videos/new.mov",
        "video_filename_clean": "new",
        "start_time": 0,
        "search_results": [],
        "last_search_query": None,
    }
