"""Submit page — pick a video on disk and queue it for ingestion."""

import os
import time
import uuid

import streamlit as st

from app.ui.path_settings import env_path_setting
from app.ui.pipeline_publish import publish_ingestion_job
from app.ui.speaker_support import supported_video_filenames

VIDEO_DATA_DIR = env_path_setting("VIDEO_DATA_PATH", "data/videos")
PROCESSED_DIR = env_path_setting("OUTPUT_DIR", "data/processed")

st.title("1 · Submit a video for ingestion")
st.caption(
    "Pick a video already staged under `"
    + VIDEO_DATA_DIR
    + "`. The worker will pick the job off RabbitMQ within a second."
)

# --- Sidebar / RabbitMQ status -----------------------------------------------

with st.sidebar:
    st.subheader("Worker queue")
    rabbitmq_url = (os.environ.get("RABBITMQ_URL") or "").strip()
    if rabbitmq_url:
        # Mask the password so a screen-share doesn't expose it.
        try:
            from urllib.parse import urlsplit, urlunsplit

            parts = urlsplit(rabbitmq_url)
            if parts.password:
                masked_netloc = (
                    f"{parts.username}:***@{parts.hostname}"
                    + (f":{parts.port}" if parts.port else "")
                )
                masked = urlunsplit(parts._replace(netloc=masked_netloc))
            else:
                masked = rabbitmq_url
        except Exception:
            masked = rabbitmq_url
        st.caption(f"`RABBITMQ_URL` = `{masked}`")
    else:
        st.error(
            "`RABBITMQ_URL` is not set. The Submit page needs RabbitMQ "
            "to enqueue jobs."
        )
        st.stop()

# --- Video picker ------------------------------------------------------------

try:
    video_files = supported_video_filenames(VIDEO_DATA_DIR)
except FileNotFoundError:
    st.error(f"Video directory `{VIDEO_DATA_DIR}` does not exist.")
    st.stop()
except ValueError as exc:
    st.error(f"Could not list videos: {exc}")
    st.stop()

if not video_files:
    st.warning(
        f"No videos found in `{VIDEO_DATA_DIR}`. Drop a file there and reload "
        "the page."
    )
    st.stop()

selected = st.selectbox(
    "Video to ingest",
    video_files,
    key="submit_selected_video",
    help="Files under " + VIDEO_DATA_DIR,
)

col1, col2 = st.columns(2)
with col1:
    title = st.text_input(
        "Title (optional)",
        help="Used by the LLM enrichment prompt for context, and by TMDb metadata lookup if `TMDB_API_KEY` is set.",
        key="submit_title",
    )
with col2:
    year = st.number_input(
        "Year (optional)",
        min_value=0,
        max_value=2100,
        value=0,
        step=1,
        format="%d",
        help="Set to 0 to omit. Used by TMDb to disambiguate titles.",
        key="submit_year",
    )

reset_outputs = st.checkbox(
    "Wipe any existing processed artifacts for this video first",
    value=False,
    key="submit_reset",
    help=(
        "Forces the worker to re-run every step (otherwise cached step "
        "outputs are reused via the cache_meta sidecars)."
    ),
)

submit_disabled = not selected
submit_clicked = st.button(
    "🚀 Publish ingestion job",
    type="primary",
    disabled=submit_disabled,
    use_container_width=True,
)

if submit_clicked:
    host_video_path = os.path.join(VIDEO_DATA_DIR, selected)
    video_filename = os.path.splitext(selected)[0]
    video_processed_dir = os.path.join(PROCESSED_DIR, video_filename)

    if reset_outputs and os.path.isdir(video_processed_dir):
        # Best-effort wipe — files written by the worker may be root-owned
        # if the worker container runs as root, which Streamlit (running
        # as a non-root container user) can't delete. We surface a warning
        # in that case but still let the demo proceed.
        try:
            import shutil

            shutil.rmtree(video_processed_dir)
            st.toast("Processed directory wiped.", icon="🧹")
        except OSError as exc:
            st.warning(
                f"Could not wipe `{video_processed_dir}` (permissions?): "
                f"{exc}. Continuing — cached steps will be reused."
            )

    with st.status("Publishing job…", expanded=True) as status:
        status.write(f"Resolving worker path for `{host_video_path}` …")
        result = publish_ingestion_job(
            host_video_path=host_video_path,
            host_video_dir=VIDEO_DATA_DIR,
            title=title or None,
            year=int(year) if year and int(year) > 0 else None,
            rabbitmq_url=rabbitmq_url,
        )
        if result.ok:
            # Mint a UI-side trace ID so the demo has something concrete to
            # show on the next page even though IngestionJob doesn't carry
            # a request_id field. Pair it with the published video path.
            demo_trace_id = uuid.uuid4().hex[:12]
            st.session_state["pipeline_video_filename"] = video_filename
            st.session_state["pipeline_video_basename"] = selected
            st.session_state["pipeline_demo_trace_id"] = demo_trace_id
            st.session_state["pipeline_started_at"] = time.time()
            st.session_state["pipeline_job_video_path"] = result.job_video_path
            status.write(f"✓ {result.detail}")
            status.write(f"Worker will look at `{result.job_video_path}`")
            status.write(f"Demo trace ID: `{demo_trace_id}`")
            status.update(label="Job published", state="complete")
        else:
            st.session_state.pop("pipeline_video_filename", None)
            status.write(f"✗ {result.detail}")
            status.update(label="Publish failed", state="error")
            st.error(result.detail)

# --- Post-submit nav hint ----------------------------------------------------

if "pipeline_video_filename" in st.session_state:
    last = st.session_state["pipeline_video_filename"]
    st.success(
        f"Last submitted: **{last}** (trace `"
        + st.session_state.get("pipeline_demo_trace_id", "?")
        + "`). Continue to **2 · Pipeline** in the sidebar to watch progress."
    )
