from typing import Any, MutableMapping


def ensure_search_session_state(state: MutableMapping[str, Any]) -> None:
    if "video_path" not in state:
        state["video_path"] = None
    if "video_filename_clean" not in state:
        state["video_filename_clean"] = None
    if "start_time" not in state:
        state["start_time"] = 0
    if "search_results" not in state:
        state["search_results"] = []


def reset_search_session_for_video(
    state: MutableMapping[str, Any],
    selected_video_file: str,
    video_path: str,
    video_filename_clean: str,
) -> None:
    if state.get("selected_video_file") == selected_video_file:
        state["video_path"] = video_path
        state["video_filename_clean"] = video_filename_clean
        return

    state["selected_video_file"] = selected_video_file
    state["video_path"] = video_path
    state["video_filename_clean"] = video_filename_clean
    state["start_time"] = 0
    state["search_results"] = []
