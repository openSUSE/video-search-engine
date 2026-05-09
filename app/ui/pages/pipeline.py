"""Pipeline page — animated step checklist for the in-flight video.

Polls ``pipeline_status()`` every ~1.5 s while any step is in_progress.
Each step renders one of:

* ⏳ pending — gray, not started yet
* 🔵 in_progress — blue, with a live elapsed-time counter
* ✅ done — green, with the wall-clock duration
* ❌ failed — red, with the failure detail

Once every step is done (or any step fails) the auto-refresh stops and a
big "Search now" CTA points the user at the Search page.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import streamlit as st

from app.ui.path_settings import env_path_setting
from app.ui.pipeline_state import (
    PIPELINE_STEPS_DISPLAY,
    PipelineStatus,
    StepStatus,
    chroma_doc_counter_factory,
    pipeline_status,
)

PROCESSED_DIR = env_path_setting("OUTPUT_DIR", "data/processed")
CHROMA_HOST = (os.environ.get("CHROMA_HOST") or "localhost").strip() or "localhost"
try:
    CHROMA_PORT = int((os.environ.get("CHROMA_PORT") or "8000").strip() or "8000")
except ValueError:
    CHROMA_PORT = 8000
CHROMA_COLLECTION = (
    os.environ.get("CHROMA_COLLECTION") or "video_search_engine"
).strip()

REFRESH_INTERVAL_SECONDS = 1.5

# ---- header -----------------------------------------------------------------

st.title("2 · Pipeline progress")

video_filename: Optional[str] = st.session_state.get("pipeline_video_filename")
video_basename: Optional[str] = st.session_state.get("pipeline_video_basename")
demo_trace_id: Optional[str] = st.session_state.get("pipeline_demo_trace_id")

if not video_filename:
    st.info(
        "No video has been submitted yet. Head to **1 · Submit** to publish "
        "an ingestion job, or pick an existing video below to inspect its "
        "current state."
    )
    # Allow inspecting any directory under data/processed/ even without a
    # fresh submit — useful for re-watching a completed run.
    if os.path.isdir(PROCESSED_DIR):
        try:
            existing = sorted(
                d
                for d in os.listdir(PROCESSED_DIR)
                if os.path.isdir(os.path.join(PROCESSED_DIR, d))
            )
        except OSError:
            existing = []
        if existing:
            picked = st.selectbox("Inspect existing processed dir", existing)
            if picked:
                video_filename = picked
                video_basename = picked

if not video_filename:
    st.stop()

processed_dir = os.path.join(PROCESSED_DIR, video_filename)

# ---- top-level summary card -------------------------------------------------

with st.container(border=True):
    summary_left, summary_right = st.columns([3, 1])
    with summary_left:
        st.subheader(f"{video_basename or video_filename}")
        st.caption(f"Processed dir: `{processed_dir}`")
        if demo_trace_id:
            st.caption(f"Demo trace ID: `{demo_trace_id}`")
    with summary_right:
        if st.button("Pick a different video", use_container_width=True):
            for key in (
                "pipeline_video_filename",
                "pipeline_video_basename",
                "pipeline_demo_trace_id",
                "pipeline_started_at",
                "pipeline_job_video_path",
            ):
                st.session_state.pop(key, None)
            st.rerun()

# ---- compute status ---------------------------------------------------------

counter = chroma_doc_counter_factory(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    collection_name=CHROMA_COLLECTION,
)


@st.cache_resource
def _resolved_processed_dir(directory: str) -> str:
    """No-op cache wrapper so Streamlit doesn't stat the directory on every
    rerun — only useful in larger trees but cheap to keep here."""
    return directory


status: PipelineStatus = pipeline_status(
    _resolved_processed_dir(processed_dir),
    video_filename,
    chroma_doc_counter=counter,
)

# ---- progress bar + headline ------------------------------------------------

progress_col, label_col = st.columns([3, 1])
with progress_col:
    st.progress(
        status.progress_fraction,
        text=f"{int(status.progress_fraction * 100)}% complete",
    )
with label_col:
    state_emoji = {
        "done": "✅",
        "in_progress": "🔵",
        "pending": "⏳",
        "failed": "❌",
    }
    st.metric(
        "Overall",
        f"{state_emoji.get(status.overall_state, '?')} {status.overall_state}",
    )

# ---- per-step list ----------------------------------------------------------


def _render_step(label: str, step: StepStatus) -> None:
    icon = {
        "done": "✅",
        "in_progress": "🔵",
        "pending": "⏳",
        "failed": "❌",
    }.get(step.state, "•")

    elapsed = step.elapsed_seconds
    if step.state == "done":
        elapsed_text = (
            f"{elapsed:.1f}s" if elapsed is not None else "—"
        )
        text_color = "#16a34a"
    elif step.state == "in_progress":
        elapsed_text = f"⏱ {elapsed:.1f}s" if elapsed is not None else "starting…"
        text_color = "#2563eb"
    elif step.state == "failed":
        elapsed_text = "failed"
        text_color = "#dc2626"
    else:
        elapsed_text = "queued"
        text_color = "#6b7280"

    cols = st.columns([0.06, 0.5, 0.2, 0.24])
    with cols[0]:
        st.markdown(f"<div style='font-size: 1.4rem'>{icon}</div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"**{label}**")
        if step.detail:
            st.caption(step.detail)
    with cols[2]:
        st.markdown(
            f"<div style='color:{text_color};font-variant-numeric:tabular-nums'>{elapsed_text}</div>",
            unsafe_allow_html=True,
        )
    with cols[3]:
        if step.finished_at:
            stamp = time.strftime("%H:%M:%S", time.localtime(step.finished_at))
            st.caption(f"finished at {stamp}")
        elif step.started_at:
            stamp = time.strftime("%H:%M:%S", time.localtime(step.started_at))
            st.caption(f"started at {stamp}")


with st.container(border=True):
    st.subheader("Pipeline steps")
    for key, label in PIPELINE_STEPS_DISPLAY:
        step = status.step(key)
        if step is None:
            continue
        _render_step(label, step)
        st.divider()

# ---- narrated current activity ---------------------------------------------

current_step = next(
    (s for s in status.steps if s.state == "in_progress"),
    None,
)
if current_step is not None:
    label = next(
        (display for key, display in PIPELINE_STEPS_DISPLAY if key == current_step.name),
        current_step.name,
    )
    st.info(
        f"**Currently running:** {label}"
        + (f" — {current_step.detail}" if current_step.detail else "")
        + (
            f" (running for {current_step.elapsed_seconds:.1f}s)"
            if current_step.elapsed_seconds
            else ""
        ),
        icon="🔵",
    )

failed_steps = [s for s in status.steps if s.state == "failed"]
if failed_steps:
    for step in failed_steps:
        label = next(
            (display for key, display in PIPELINE_STEPS_DISPLAY if key == step.name),
            step.name,
        )
        st.error(
            f"**{label} failed**" + (f": {step.detail}" if step.detail else ""),
            icon="❌",
        )

# ---- post-completion CTA ----------------------------------------------------

if status.overall_state == "done":
    st.success(
        "Pipeline complete — every step succeeded. Continue to **3 · Search** "
        "to query the indexed segments.",
        icon="🎉",
    )
    if st.button("Open Search ▶", type="primary", use_container_width=True):
        st.switch_page(
            os.path.join(os.path.dirname(__file__), "search.py")
        )

# ---- auto-refresh while in progress ----------------------------------------

# st.rerun() in a tight loop with a sleep is the simplest pattern Streamlit
# offers for live polling. We only refresh while any step is non-terminal,
# so a completed pipeline doesn't burn CPU.
if status.overall_state in {"in_progress", "pending"}:
    time.sleep(REFRESH_INTERVAL_SECONDS)
    st.rerun()
