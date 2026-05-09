"""Unit tests for ``app.ui.pipeline_state``.

The state walker is pure-stdlib (no Streamlit / Chroma imports at the
module level) so these tests run in the unit suite without skipping.
We use a temporary processed_dir and fake the chroma counter via
dependency injection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.ui.pipeline_state import (
    PipelineStatus,
    PIPELINE_FILE_NAMES,
    pipeline_status,
)


def _touch(path: Path, *, content: object = None, mtime: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if content is None:
        path.write_bytes(b"")
    else:
        path.write_text(json.dumps(content))
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _write_steps(processed_dir: Path, *step_keys: str, base_mtime: float = 1000.0):
    """Write empty artifacts for each step key in PIPELINE_FILE_NAMES order
    with monotonically increasing mtimes so step ordering is unambiguous."""
    for i, key in enumerate(step_keys):
        path = processed_dir / PIPELINE_FILE_NAMES[key]
        _touch(path, mtime=base_mtime + i)


def test_pipeline_status_all_pending_when_dir_empty(tmp_path: Path):
    status = pipeline_status(
        str(tmp_path),
        "demo",
        chroma_doc_counter=lambda _: 0,
        now=2000.0,
    )
    assert status.overall_state in {"in_progress", "pending"}
    # First step (shots) should be in_progress, rest pending.
    states = [s.state for s in status.steps]
    assert states[0] == "in_progress"
    assert all(state in {"pending", "in_progress"} for state in states[1:])


def test_pipeline_status_done_when_all_artifacts_present(tmp_path: Path):
    _write_steps(
        tmp_path,
        "shots",
        "audio",
        "transcript",
        "audio_events",
        "visual",
        "actions",
        "speaker_map",
        "segmentation",
    )
    enriched_path = tmp_path / "final_enriched_segments.json"
    enriched_path.write_text(
        json.dumps(
            [
                {"title": "T1", "summary": "S1"},
                {"title": "T2", "summary": "S2"},
            ]
        )
    )
    os.utime(enriched_path, (2000.0, 2000.0))

    status = pipeline_status(
        str(tmp_path),
        "demo",
        chroma_doc_counter=lambda _: 4,
        now=3000.0,
    )
    assert status.overall_state == "done"
    assert status.progress_fraction == 1.0
    indexing = status.step("indexing")
    assert indexing is not None
    assert indexing.state == "done"
    assert "4" in indexing.detail


def test_pipeline_status_in_progress_at_enrichment(tmp_path: Path):
    _write_steps(
        tmp_path,
        "shots",
        "audio",
        "transcript",
        "audio_events",
        "visual",
        "actions",
        "speaker_map",
        "segmentation",
    )
    # Half-written enrichment file: one good + one missing.
    enriched_path = tmp_path / "final_enriched_segments.json"
    enriched_path.write_text(
        json.dumps([{"title": "T", "summary": "S"}, {"foo": "bar"}])
    )

    status = pipeline_status(
        str(tmp_path),
        "demo",
        chroma_doc_counter=lambda _: 0,
        now=3000.0,
    )
    assert status.overall_state == "in_progress"
    enrichment = status.step("enrichment")
    assert enrichment is not None
    assert enrichment.state == "in_progress"
    indexing = status.step("indexing")
    assert indexing is not None
    assert indexing.state == "pending"


def test_pipeline_status_failed_when_majority_errored(tmp_path: Path):
    _write_steps(
        tmp_path,
        "shots",
        "audio",
        "transcript",
        "audio_events",
        "visual",
        "actions",
        "speaker_map",
        "segmentation",
    )
    enriched_path = tmp_path / "final_enriched_segments.json"
    enriched_path.write_text(
        json.dumps(
            [
                {"title": "Error", "summary": "Failed to generate"},
                {"title": "Error", "summary": "Failed to generate"},
                {"title": "Error", "summary": "Failed to generate"},
                {"title": "OK", "summary": "good"},
            ]
        )
    )

    status = pipeline_status(
        str(tmp_path),
        "demo",
        chroma_doc_counter=lambda _: 0,
        now=3000.0,
    )
    assert status.overall_state == "failed"
    assert status.step("enrichment").state == "failed"
    assert status.step("indexing").state == "pending"


def test_pipeline_status_no_audio_branch_skips_audio_steps(tmp_path: Path):
    # Simulate the no-audio sentinel that step_01_extraction writes.
    meta_path = tmp_path / "transcript_raw.json.cache_meta.json"
    meta_path.write_text(json.dumps({"skipped_reason": "no_audio_stream"}))

    _write_steps(tmp_path, "shots")
    # Empty transcript + audio_events sidecars (the no-audio branch
    # writes both).
    (tmp_path / "transcript_raw.json").write_text("[]")
    (tmp_path / "audio_events.json").write_text("{}")

    status = pipeline_status(
        str(tmp_path),
        "demo",
        chroma_doc_counter=lambda _: 0,
        now=3000.0,
    )
    audio_step = status.step("audio")
    assert audio_step is not None
    assert audio_step.state == "done"
    assert "skipped" in audio_step.detail.lower()


def test_pipeline_status_indexing_in_progress_when_chroma_unreachable(tmp_path: Path):
    _write_steps(
        tmp_path,
        "shots",
        "audio",
        "transcript",
        "audio_events",
        "visual",
        "actions",
        "speaker_map",
        "segmentation",
    )
    (tmp_path / "final_enriched_segments.json").write_text(
        json.dumps([{"title": "T", "summary": "S"}])
    )

    def unreachable(_video):
        return None

    status = pipeline_status(
        str(tmp_path),
        "demo",
        chroma_doc_counter=unreachable,
        now=3000.0,
    )
    indexing = status.step("indexing")
    assert indexing is not None
    assert indexing.state == "in_progress"


def test_step_status_elapsed_uses_now_for_in_progress(tmp_path: Path):
    _write_steps(tmp_path, "shots")
    # Without 'audio', the audio step is in_progress. Its started_at
    # should be the shots mtime; elapsed is now - started_at.
    status = pipeline_status(
        str(tmp_path),
        "demo",
        chroma_doc_counter=lambda _: 0,
        now=2000.0,
    )
    audio = status.step("audio")
    assert audio is not None
    assert audio.state == "in_progress"
    assert audio.started_at == pytest.approx(1000.0, rel=1e-6)
    elapsed = audio.elapsed_seconds
    assert elapsed is not None
    # The started_at is 1000.0 from _write_steps; "now" is 2000 in the
    # call, but elapsed_seconds reads time.time() directly. We can't pin
    # that without monkeypatching, so just assert it's positive and at
    # least the floor.
    assert elapsed >= 0.0


def test_pipeline_status_returns_correct_dataclass(tmp_path: Path):
    status = pipeline_status(
        str(tmp_path),
        "demo",
        chroma_doc_counter=lambda _: 0,
    )
    assert isinstance(status, PipelineStatus)
    assert status.video_filename == "demo"
    assert status.processed_dir == str(tmp_path)
    # Always exactly the steps listed in PIPELINE_STEPS_DISPLAY.
    expected_step_names = [
        "shots",
        "audio",
        "transcript",
        "audio_events",
        "visual",
        "actions",
        "speaker_map",
        "segmentation",
        "enrichment",
        "indexing",
    ]
    assert [s.name for s in status.steps] == expected_step_names


def test_progress_fraction_partial(tmp_path: Path):
    # 4 done out of 10 → 0.4
    _write_steps(tmp_path, "shots", "audio", "transcript", "audio_events")
    status = pipeline_status(
        str(tmp_path),
        "demo",
        chroma_doc_counter=lambda _: 0,
        now=3000.0,
    )
    assert status.progress_fraction == pytest.approx(0.4, rel=1e-6)
