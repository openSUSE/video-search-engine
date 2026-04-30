import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from core.atomic_io import atomic_write_json

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


def is_supported_video_file(filename: str | Path) -> bool:
    path = Path(filename)
    return bool(path.suffix and path.stem and path.suffix.lower() in VIDEO_EXTENSIONS)


def supported_video_filenames(video_data_dir: str | Path) -> list[str]:
    video_dir = Path(video_data_dir)
    return sorted(
        path.name
        for path in video_dir.iterdir()
        if path.is_file() and is_supported_video_file(path.name)
    )


def resolve_video_path(
    video_data_dir: str | Path,
    video_stem: str,
    extensions: Sequence[str] = VIDEO_EXTENSIONS,
) -> Path:
    video_dir = Path(video_data_dir)
    for extension in extensions:
        candidate = video_dir / f"{video_stem}{extension}"
        if candidate.is_file():
            return candidate
        if not video_dir.is_dir():
            continue
        candidate_name = candidate.name.lower()
        for existing_path in video_dir.iterdir():
            if existing_path.is_file() and existing_path.name.lower() == candidate_name:
                return existing_path
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


def speaker_segment_button_key(speaker_label: str, segment_index: int) -> str:
    return f"seg_{speaker_label}_{segment_index}"


def validate_speaker_ids_from_transcript(transcript_segments: Any) -> list[str]:
    if not isinstance(transcript_segments, list):
        raise ValueError("transcript must be a JSON array")

    speaker_ids = set()
    for segment_index, segment in enumerate(transcript_segments):
        if not isinstance(segment, Mapping):
            raise ValueError(
                f"transcript segment at index {segment_index} must be a JSON object"
            )

        speaker_id = segment.get("speaker")
        if speaker_id is None:
            continue
        if not isinstance(speaker_id, str):
            raise ValueError(
                f"transcript segment at index {segment_index} must have string speaker"
            )
        if speaker_id.strip():
            speaker_ids.add(speaker_id.strip())

    return sorted(speaker_ids)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _transcript_time_value(
    segment: Mapping[str, Any],
    segment_index: int,
    field_name: str,
) -> float:
    value = segment.get(field_name)
    if not _is_number(value):
        raise ValueError(
            f"transcript segment at index {segment_index} must have numeric {field_name}"
        )
    if value < 0:
        raise ValueError(
            f"transcript segment at index {segment_index} must have non-negative {field_name}"
        )
    return float(value)


def validate_transcript_segments_for_display(
    transcript_segments: Any,
) -> list[dict[str, Any]]:
    if not isinstance(transcript_segments, list):
        raise ValueError("transcript must be a JSON array")

    normalized_segments = []
    for segment_index, segment in enumerate(transcript_segments):
        if not isinstance(segment, Mapping):
            raise ValueError(
                f"transcript segment at index {segment_index} must be a JSON object"
            )

        start = _transcript_time_value(segment, segment_index, "start")
        end = _transcript_time_value(segment, segment_index, "end")
        if end < start:
            raise ValueError(
                f"transcript segment at index {segment_index} "
                "end must be greater than or equal to start"
            )

        if not isinstance(segment.get("text"), str):
            raise ValueError(
                f"transcript segment at index {segment_index} must have string text"
            )

        normalized_segment = dict(segment)
        normalized_segment["start"] = start
        normalized_segment["end"] = end

        speaker = segment.get("speaker")
        if speaker is not None:
            if not isinstance(speaker, str):
                raise ValueError(
                    f"transcript segment at index {segment_index} "
                    "must have string speaker"
                )
            normalized_segment["speaker"] = speaker.strip()

        normalized_segments.append(normalized_segment)

    return normalized_segments


def load_transcript_segments(path: str | Path) -> list[dict[str, Any]]:
    try:
        with Path(path).open("r") as f:
            transcript_segments = json.load(f)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"transcript file could not be read: {exc}") from exc
    return validate_transcript_segments_for_display(transcript_segments)


def load_transcript_speaker_ids(path: str | Path) -> list[str]:
    return validate_speaker_ids_from_transcript(load_transcript_segments(path))


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


def load_speaker_map(path: str | Path) -> dict[str, str]:
    try:
        with Path(path).open("r") as f:
            speaker_map = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return normalize_speaker_map(speaker_map) or {}


def save_speaker_map_if_complete(
    path: str | Path,
    speaker_map: Any,
    transcript_speaker_ids: Sequence[str],
) -> bool:
    normalized_speaker_map = normalize_speaker_map(speaker_map)
    if normalized_speaker_map is None:
        return False

    required_speaker_ids = {
        speaker_id.strip()
        for speaker_id in transcript_speaker_ids
        if isinstance(speaker_id, str) and speaker_id.strip()
    }
    if required_speaker_ids - set(normalized_speaker_map):
        return False

    atomic_write_json(path, normalized_speaker_map)
    return True


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
