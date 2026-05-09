"""Read-only snapshot of where a video is in the ingestion pipeline.

The Streamlit "pipeline" page polls ``pipeline_status()`` to render an
animated step checklist. Status is derived purely from artifacts on disk
plus an optional Chroma docs-count probe, so we don't need a side channel
to the worker, don't need to parse logs, and don't need any server-state
schema changes. The same function works for the local-pipeline path
(``run_pipeline.py``) and the queue-driven worker.

Steps:

* shots (shots.json)
* audio (normalized_audio.mp3) — skipped iff the no-audio sentinel is set
* transcript (transcript_raw.json)
* audio_events (audio_events.json)
* visual (visual_details.json)
* actions (actions.json)
* speaker_map (speaker_map.json)
* segmentation (final_segments.json)
* enrichment (final_enriched_segments.json)
* indexing (Chroma collection has rows for video_filename)

A step is:

* ``"pending"`` until any prior step is in_progress / pending
* ``"in_progress"`` when prior steps are all ``"done"`` but the current
  artifact does not yet exist
* ``"done"`` when the artifact exists (and, for enrichment + indexing,
  the per-segment validation passes)
* ``"failed"`` when the artifact exists but is structurally broken
  (e.g. ``"title": "Error"`` for >20% of enriched segments)

Wall-clock elapsed times use file mtimes: the started_at of step N is
set to the latest mtime among steps 0..N-1, and the finished_at of step
N is the mtime of step N's artifact.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


PIPELINE_FILE_NAMES = {
    "shots": "shots.json",
    "audio": "normalized_audio.mp3",
    "transcript": "transcript_raw.json",
    "audio_events": "audio_events.json",
    "visual": "visual_details.json",
    "actions": "actions.json",
    "speaker_map": "speaker_map.json",
    "segmentation": "final_segments.json",
    "enrichment": "final_enriched_segments.json",
}

# Display order + human labels (used by the Streamlit page so this module
# stays the single source of truth for step ordering).
PIPELINE_STEPS_DISPLAY: list[tuple[str, str]] = [
    ("shots", "Shot detection"),
    ("audio", "Audio extraction"),
    ("transcript", "Transcription + diarization"),
    ("audio_events", "Audio events"),
    ("visual", "Visual captions"),
    ("actions", "Action recognition"),
    ("speaker_map", "Speaker map"),
    ("segmentation", "Segmentation"),
    ("enrichment", "LLM enrichment"),
    ("indexing", "Indexing into Chroma"),
]


@dataclass
class StepStatus:
    name: str
    state: str  # "pending" | "in_progress" | "done" | "failed"
    started_at: Optional[float] = None  # epoch seconds
    finished_at: Optional[float] = None  # epoch seconds
    detail: str = ""

    @property
    def elapsed_seconds(self) -> Optional[float]:
        """Wall-clock seconds spent on this step.

        Returns None for pending steps (haven't started yet). For an
        in-progress step we measure against now() so the UI shows a live
        timer; for done/failed steps we use the recorded finish time.
        """
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)


@dataclass
class PipelineStatus:
    video_filename: str
    processed_dir: str
    steps: List[StepStatus] = field(default_factory=list)

    @property
    def overall_state(self) -> str:
        """Aggregate state for the whole pipeline."""
        states = {step.state for step in self.steps}
        if "failed" in states:
            return "failed"
        if "in_progress" in states:
            return "in_progress"
        if states == {"done"}:
            return "done"
        return "pending"

    @property
    def progress_fraction(self) -> float:
        """0.0 to 1.0 over all steps. Used for a top-level progress bar."""
        if not self.steps:
            return 0.0
        return sum(1 for step in self.steps if step.state == "done") / len(self.steps)

    def step(self, name: str) -> Optional[StepStatus]:
        for entry in self.steps:
            if entry.name == name:
                return entry
        return None


def _safe_mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _enrichment_outcome(enriched_path: str) -> tuple[str, str]:
    """Return (state, detail) by reading the partially-written file.

    "done" when every segment has a non-Error title.
    "in_progress" when some segments are still missing title/summary.
    "failed" when >20% of segments are stuck on title=="Error".

    Mirrors the same threshold used in ``step_03_enrichment.py``.
    """
    try:
        with open(enriched_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return ("failed", f"unreadable: {exc}")
    if not isinstance(data, list):
        return ("failed", "expected a JSON array")
    total = len(data)
    if total == 0:
        return ("failed", "empty segment list")
    error_count = sum(
        1
        for seg in data
        if isinstance(seg, dict) and seg.get("title") == "Error"
    )
    missing_count = sum(
        1
        for seg in data
        if not isinstance(seg, dict)
        or not isinstance(seg.get("title"), str)
        or not isinstance(seg.get("summary"), str)
    )
    enriched_count = total - error_count - missing_count
    if missing_count == 0 and error_count == 0:
        return ("done", f"{total}/{total} enriched")
    if missing_count > 0:
        return (
            "in_progress",
            f"{enriched_count}/{total} enriched, {missing_count} pending",
        )
    # Some segments are "Error" but no longer pending; mirror the runtime
    # 20%-failure tolerance — softer than a hard fail, harder than success.
    if error_count / total > 0.20:
        return ("failed", f"{error_count}/{total} stuck on Error")
    return ("done", f"{enriched_count}/{total} enriched, {error_count} skipped")


def _has_audio_marker(processed_dir: str) -> Optional[bool]:
    """Return True if we know the video has audio, False if no-audio sentinel
    was written, or None if we don't know yet (pre-extraction).

    The extraction step writes ``transcript_raw.json.cache_meta.json`` with
    ``skipped_reason: no_audio_stream`` when the silent-video branch fires.
    """
    meta_path = os.path.join(processed_dir, "transcript_raw.json.cache_meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data.get("skipped_reason") != "no_audio_stream"


# Default Chroma probe: counts docs scoped to a video_filename. Returns
# None when chromadb isn't installed or the server is unreachable so the
# UI can degrade gracefully instead of erroring out.
def _default_chroma_count(
    video_filename: str,
    *,
    host: str,
    port: int,
    collection_name: str,
) -> Optional[int]:
    try:
        import chromadb
    except ImportError:
        return None
    try:
        client = chromadb.HttpClient(host=host, port=port)
        client.heartbeat()
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        result = collection.get(
            where={"video_filename": video_filename},
            include=[],
            limit=1,
        )
    except Exception:
        return None
    ids = result.get("ids") if isinstance(result, dict) else None
    return None if ids is None else len(_collect_all_ids(collection, video_filename))


def _collect_all_ids(collection: Any, video_filename: str) -> List[str]:
    try:
        result = collection.get(
            where={"video_filename": video_filename},
            include=[],
        )
    except Exception:
        return []
    ids = result.get("ids") if isinstance(result, dict) else None
    return list(ids) if isinstance(ids, list) else []


def pipeline_status(
    processed_dir: str,
    video_filename: str,
    *,
    chroma_doc_counter: Optional[Callable[[str], Optional[int]]] = None,
    has_audio_override: Optional[bool] = None,
    now: Optional[float] = None,
) -> PipelineStatus:
    """Build a snapshot of the per-step pipeline state for ``video_filename``.

    Parameters
    ----------
    processed_dir
        The directory the worker writes artifacts into
        (``<OUTPUT_DIR>/<video_filename>``).
    video_filename
        The stem of the video, used as the Chroma metadata scope.
    chroma_doc_counter
        Optional injected callable that maps ``video_filename -> int``.
        Returning None means "Chroma is unreachable; treat indexing as
        in_progress when prior steps are done." Tests pass a stub here
        so the function is unit-testable without a running Chroma.
    has_audio_override
        Test hook — bypass the cache_meta sniffing and force the audio
        branch decision.
    now
        Test hook — pin the wall-clock for deterministic elapsed-time
        assertions.
    """
    now_ts = now if now is not None else time.time()

    artifact_paths = {
        key: os.path.join(processed_dir, fname)
        for key, fname in PIPELINE_FILE_NAMES.items()
    }
    has_audio: Optional[bool]
    if has_audio_override is not None:
        has_audio = has_audio_override
    else:
        has_audio = _has_audio_marker(processed_dir)

    steps: list[StepStatus] = []

    def add(name: str, state: str, *, started_at=None, finished_at=None, detail=""):
        steps.append(
            StepStatus(
                name=name,
                state=state,
                started_at=started_at,
                finished_at=finished_at,
                detail=detail,
            )
        )

    # We track the "running tail" mtime to set started_at on the next step.
    # When a step is "done" its finished_at is the artifact mtime; the next
    # step inherits it as its started_at (best-effort, since the worker
    # doesn't emit per-step start events).
    last_finished_at: Optional[float] = None
    blocked = False  # once a step is pending/in_progress, downstream steps stay pending.

    def file_step(name: str, path: str) -> None:
        nonlocal last_finished_at, blocked
        mtime = _safe_mtime(path)
        if blocked or mtime is None:
            if blocked:
                add(name, "pending")
            else:
                # First not-yet-existing artifact in the chain → in_progress.
                add(name, "in_progress", started_at=last_finished_at)
                blocked = True
            return
        add(name, "done", started_at=last_finished_at, finished_at=mtime)
        last_finished_at = mtime

    def skipped_step(name: str, detail: str) -> None:
        nonlocal last_finished_at
        # A skipped step is "done" with zero elapsed time; downstream
        # steps treat it as a no-op.
        ts = last_finished_at or now_ts
        add(name, "done", started_at=ts, finished_at=ts, detail=detail)

    file_step("shots", artifact_paths["shots"])

    if has_audio is False:
        skipped_step("audio", "skipped — no audio stream")
        # Transcript and audio events are written as empty sidecars by
        # the extraction step in the no-audio branch, so we still file_step
        # them — they'll resolve to "done" near-instantaneously.
        file_step("transcript", artifact_paths["transcript"])
        file_step("audio_events", artifact_paths["audio_events"])
    else:
        file_step("audio", artifact_paths["audio"])
        file_step("transcript", artifact_paths["transcript"])
        file_step("audio_events", artifact_paths["audio_events"])

    file_step("visual", artifact_paths["visual"])
    file_step("actions", artifact_paths["actions"])
    file_step("speaker_map", artifact_paths["speaker_map"])
    file_step("segmentation", artifact_paths["segmentation"])

    # Enrichment is more nuanced — the file can exist while still being
    # half-filled, so we open it and look at per-segment state.
    enriched_path = artifact_paths["enrichment"]
    if blocked or not os.path.exists(enriched_path):
        if blocked:
            add("enrichment", "pending")
        else:
            add("enrichment", "in_progress", started_at=last_finished_at)
            blocked = True
    else:
        outcome, detail = _enrichment_outcome(enriched_path)
        mtime = _safe_mtime(enriched_path)
        if outcome == "done":
            add(
                "enrichment",
                "done",
                started_at=last_finished_at,
                finished_at=mtime,
                detail=detail,
            )
            last_finished_at = mtime
        elif outcome == "failed":
            add(
                "enrichment",
                "failed",
                started_at=last_finished_at,
                finished_at=mtime,
                detail=detail,
            )
            blocked = True
        else:
            add(
                "enrichment",
                "in_progress",
                started_at=last_finished_at,
                detail=detail,
            )
            blocked = True

    # Indexing — count Chroma rows for this video. None means unreachable.
    if blocked:
        add("indexing", "pending")
    elif chroma_doc_counter is None:
        # No probe configured; treat as in_progress so the user can tell
        # the rest of the pipeline succeeded.
        add(
            "indexing",
            "in_progress",
            started_at=last_finished_at,
            detail="Chroma probe not configured",
        )
    else:
        try:
            count = chroma_doc_counter(video_filename)
        except Exception as exc:
            count = None
            detail = f"chroma probe error: {exc}"
        else:
            detail = ""
        if count is None:
            add(
                "indexing",
                "in_progress",
                started_at=last_finished_at,
                detail=detail or "Chroma unreachable",
            )
        elif count == 0:
            add(
                "indexing",
                "in_progress",
                started_at=last_finished_at,
                detail="0 documents indexed yet",
            )
        else:
            add(
                "indexing",
                "done",
                started_at=last_finished_at,
                finished_at=now_ts,
                detail=f"{count} documents",
            )

    return PipelineStatus(
        video_filename=video_filename,
        processed_dir=processed_dir,
        steps=steps,
    )


def chroma_doc_counter_factory(
    *,
    host: str,
    port: int,
    collection_name: str,
) -> Callable[[str], Optional[int]]:
    """Bind connection details into a counter the UI can pass in.

    The factory exists so the Streamlit page doesn't have to know about
    chromadb's HttpClient signature directly — it just calls
    ``counter(video_filename)``.
    """

    def counter(video_filename: str) -> Optional[int]:
        return _default_chroma_count(
            video_filename,
            host=host,
            port=port,
            collection_name=collection_name,
        )

    return counter
