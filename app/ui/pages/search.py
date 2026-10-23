"""Search page — query the indexed segments and jump the player to hits.

Polished demo flavor over the original `search_app.py`:

* Sample queries as clickable chips that pre-populate the input.
* The X-Request-ID returned by the API is shown next to results so the
  audience can correlate with `kubectl logs | grep req=...`.
* A "Reset demo" button drops the Chroma collection so the next demo
  starts from a clean slate.
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st

from app.ui.path_settings import env_path_setting
from app.ui.search_client import (
    RequestException,
    format_time_range,
    post_search,
    search_api_url,
    search_payload,
    search_result_play_button_key,
    search_results_from_response,
)
from app.ui.search_state import (
    ensure_search_session_state,
    reset_search_session_for_video,
)
from app.ui.speaker_support import supported_video_filenames

VIDEO_DATA_DIR = env_path_setting("VIDEO_DATA_PATH", "data/videos")
API_URL = search_api_url()

ensure_search_session_state(st.session_state)

# ---- Sidebar: video picker + reset -----------------------------------------

with st.sidebar:
    st.header("Video Selection")
    try:
        video_files = supported_video_filenames(VIDEO_DATA_DIR)
    except FileNotFoundError:
        st.error(f"Video directory not found at `{VIDEO_DATA_DIR}`.")
        st.stop()
    except ValueError as exc:
        st.error(f"Ambiguous video selection: {exc}")
        st.stop()

    if not video_files:
        st.warning(f"No video files found in `{VIDEO_DATA_DIR}`.")
        st.stop()

    # If we just submitted, default to that video for continuity.
    submitted = st.session_state.get("pipeline_video_basename")
    initial_index = 0
    if submitted in video_files:
        initial_index = video_files.index(submitted)

    selected_video_file = st.selectbox(
        "Choose a video to search:", video_files, index=initial_index
    )
    reset_search_session_for_video(
        st.session_state,
        selected_video_file,
        os.path.join(VIDEO_DATA_DIR, selected_video_file),
        os.path.splitext(selected_video_file)[0],
    )

    top_k = st.slider("Number of results", min_value=1, max_value=50, value=5)

    st.divider()
    st.subheader("Demo controls")
    st.caption(
        "Drop the Chroma collection — useful between demo runs so the "
        "audience watches the data populate live on the next ingestion."
    )
    if st.button("🧹 Reset Chroma collection", use_container_width=True):
        try:
            import chromadb

            host = (os.environ.get("CHROMA_HOST") or "localhost").strip()
            port = int((os.environ.get("CHROMA_PORT") or "8000").strip())
            collection_name = (
                os.environ.get("CHROMA_COLLECTION") or "video_search_engine"
            ).strip()
            client = chromadb.HttpClient(host=host, port=port)
            try:
                client.delete_collection(collection_name)
                st.toast(f"Dropped collection {collection_name!r}.", icon="🧹")
            except Exception as exc:
                # delete_collection raises when the collection doesn't
                # already exist; in demo terms that's still success.
                st.toast(f"Collection already empty ({exc}).", icon="ℹ️")
        except ImportError:
            st.error("chromadb is not available in this image.")

# ---- Title + sample queries -------------------------------------------------

st.title("3 · Search")
st.caption(
    "Hybrid retrieval (text + visual modalities, RRF-fused) with "
    "duration filtering and per-video scoping."
)

SAMPLE_QUERIES_BY_VIDEO = {
    "sintel_trailer": [
        "young woman fighting a dragon",
        "snowy mountain landscape",
        "dangerous quest",
        "desert travel",
    ],
}

stem = st.session_state.video_filename_clean or ""
sample_queries = SAMPLE_QUERIES_BY_VIDEO.get(stem, [])

if sample_queries:
    st.markdown("**Try one of these:**")
    chip_cols = st.columns(min(4, len(sample_queries)))
    for i, sample in enumerate(sample_queries):
        with chip_cols[i % len(chip_cols)]:
            if st.button(sample, key=f"chip-{i}", use_container_width=True):
                st.session_state["search_query_input"] = sample

# ---- Main area: query + player ---------------------------------------------

col_q, col_v = st.columns([1, 1.2])

with col_q:
    st.subheader("Query")
    query = (
        st.text_input(
            "What are you looking for?",
            placeholder="e.g., a man holding a gun",
            max_chars=1000,
            key="search_query_input",
        )
        or ""
    )
    query = query.strip()

    duration_col1, duration_col2 = st.columns(2)
    with duration_col1:
        min_duration = st.number_input(
            "Min duration (s)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            help="Drop segments shorter than this from the results (P2-22 filter).",
        )
    with duration_col2:
        max_duration = st.number_input(
            "Max duration (s)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            help="Drop segments longer than this. 0 means no upper bound.",
        )

    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching for relevant moments…"):
                payload = search_payload(
                    query,
                    st.session_state.video_filename_clean,
                    top_k=top_k,
                )
                if min_duration > 0:
                    payload["min_duration_sec"] = float(min_duration)
                if max_duration > 0:
                    payload["max_duration_sec"] = float(max_duration)
                try:
                    response = post_search(API_URL, payload)
                    if response.status_code == 200:
                        try:
                            st.session_state.search_results = (
                                search_results_from_response(response)
                            )
                            st.session_state.last_search_query = query
                            # Pull the request_id back so the demo can
                            # quote it. requests treats headers
                            # case-insensitively.
                            st.session_state["last_request_id"] = response.headers.get(
                                "x-request-id"
                            )
                        except ValueError as exc:
                            st.error(f"Search API returned an unusable response: {exc}")
                            st.session_state.search_results = []
                            st.session_state.last_search_query = None
                    else:
                        st.error(
                            f"Failed to get results from API "
                            f"(status {response.status_code}).\n\n"
                            f"Response: {response.text}"
                        )
                        st.session_state.search_results = []
                        st.session_state.last_search_query = None
                except RequestException as exc:
                    st.error(
                        f"Could not connect to the Search API at {API_URL}. "
                        f"Is it running?\n\nDetails: {exc}"
                    )
                    st.session_state.search_results = []
                    st.session_state.last_search_query = None

with col_v:
    st.subheader("Player")
    if st.session_state.video_path:
        st.video(
            st.session_state.video_path,
            start_time=st.session_state.start_time,
        )
    else:
        st.info("Select a video from the sidebar to begin.")

if st.session_state.get("last_request_id"):
    st.caption(
        f"Last request traced as `req={st.session_state['last_request_id']}` "
        "— grep that across api/worker logs to follow the call end-to-end."
    )

st.divider()

# ---- Results ---------------------------------------------------------------

st.subheader("Results")
results = st.session_state.search_results
last_query = st.session_state.last_search_query
if results:
    for result_index, result in enumerate(results):
        with st.container(border=True):
            res_col1, res_col2 = st.columns([4, 1])
            with res_col1:
                st.markdown(f"**Title.** {result['title']}")
                st.write(result["summary"])
                time_range = format_time_range(
                    result["start_time"], result["end_time"]
                )
                duration = result["end_time"] - result["start_time"]
                st.caption(
                    f"⏱ {time_range} · ⌛ {duration:.1f}s · 🗣 "
                    f"{result['speakers'] or 'N/A'} · 📊 "
                    f"score {result['score']:.4f}"
                )
            with res_col2:
                if st.button(
                    "▶️ Jump to start",
                    key=search_result_play_button_key(result["id"], result_index),
                    use_container_width=True,
                ):
                    st.session_state.start_time = int(result["start_time"])
                    st.rerun()
elif last_query:
    st.info(f"No matches found for {last_query!r}.")
else:
    st.info("Your search results will appear here.")
