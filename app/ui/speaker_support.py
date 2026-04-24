from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

DEFAULT_TRANSCRIPT_FILENAME = "transcript_generic.json"
DEFAULT_SPEAKER_MAP_FILENAME = "speaker_map.json"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi")


@dataclass(frozen=True)
class SpeakerArtifactPaths:
    video_dir: Path
    transcript: Path
    video: Path
    speaker_map: Path


def processed_video_folders(base_dir: str | Path) -> list[str]:
    base_path = Path(base_dir)
    return sorted(path.name for path in base_path.iterdir() if path.is_dir())


def resolve_video_path(
    video_data_dir: str | Path,
    video_stem: str,
    extensions: Sequence[str] = VIDEO_EXTENSIONS,
) -> Path:
    video_dir = Path(video_data_dir)
    for extension in extensions:
        candidate = video_dir / f"{video_stem}{extension}"
        if candidate.exists():
            return candidate
    return video_dir / f"{video_stem}{extensions[0]}"


def speaker_artifact_paths(
    base_dir: str | Path,
    video_folder: str,
    video_data_dir: str | Path,
    config: Mapping[str, Any],
) -> SpeakerArtifactPaths:
    filenames = config.get("filenames", {})
    video_specific_dir = Path(base_dir) / video_folder
    transcript_name = filenames.get("transcript", DEFAULT_TRANSCRIPT_FILENAME)
    speaker_map_name = filenames.get("speaker_map", DEFAULT_SPEAKER_MAP_FILENAME)

    return SpeakerArtifactPaths(
        video_dir=video_specific_dir,
        transcript=video_specific_dir / transcript_name,
        video=resolve_video_path(video_data_dir, video_folder),
        speaker_map=video_specific_dir / speaker_map_name,
    )


def speaker_ids_from_transcript(transcript_segments: Any) -> list[str]:
    if not isinstance(transcript_segments, list):
        return []

    speaker_ids = set()
    for segment in transcript_segments:
        if not isinstance(segment, Mapping):
            continue
        speaker_id = segment.get("speaker")
        if isinstance(speaker_id, str) and speaker_id.strip():
            speaker_ids.add(speaker_id.strip())

    return sorted(speaker_ids)


def normalize_speaker_map(raw_speaker_map: Any) -> dict[str, str] | None:
    if not isinstance(raw_speaker_map, dict):
        return None

    speaker_map = {}
    for speaker_id, speaker_name in raw_speaker_map.items():
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            return None
        if not isinstance(speaker_name, str) or not speaker_name.strip():
            return None

        normalized_speaker_id = speaker_id.strip()
        if normalized_speaker_id in speaker_map:
            return None
        speaker_map[normalized_speaker_id] = speaker_name.strip()

    return speaker_map


def ensure_speaker_session_state(state: MutableMapping[str, Any]) -> None:
    if "speaker_map" not in state:
        state["speaker_map"] = {}
    if "video_start_time" not in state:
        state["video_start_time"] = 0
    if "current_transcript_data" not in state:
        state["current_transcript_data"] = None


def reset_speaker_session_for_video(
    state: MutableMapping[str, Any],
    selected_video_folder: str,
    base_dir: str | Path | None = None,
) -> None:
    selection_key = (
        f"{Path(base_dir)}::{selected_video_folder}"
        if base_dir is not None
        else selected_video_folder
    )
    if state.get("selected_video_key") == selection_key:
        return

    state["selected_video_key"] = selection_key
    state["selected_video_folder"] = selected_video_folder
    state["speaker_map"] = {}
    state["video_start_time"] = 0
    state["current_transcript_data"] = None
