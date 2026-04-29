# app/ui/search_app.py

import os
import sys

import streamlit as st

# Add the project root to the Python path to allow importing from 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.ui.path_settings import env_path_setting
from app.ui.search_client import (
    RequestException,
    format_time_range,
    post_search,
    search_api_url,
    search_payload,
    search_results_from_response,
)
from app.ui.search_state import (
    ensure_search_session_state,
    reset_search_session_for_video,
)

VIDEO_DATA_DIR = env_path_setting("VIDEO_DATA_PATH", "data/videos")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Semantic Video Search",
    layout="wide"
)

API_URL = search_api_url()

# --- SESSION STATE INITIALIZATION ---
# Session state holds variables that persist across user interactions.
ensure_search_session_state(st.session_state)

# --- UI LAYOUT ---
st.title("🎬 Semantic Video Search Engine")
st.markdown("Search through your videos by describing what was said or what was shown.")

# --- SIDEBAR FOR VIDEO SELECTION ---
with st.sidebar:
    st.header("Video Selection")
    
    try:
        video_files = [f for f in os.listdir(VIDEO_DATA_DIR) if f.endswith(('.mp4', '.mov', '.avi'))]
        if not video_files:
            st.warning(f"No video files found in '{VIDEO_DATA_DIR}'.")
            st.stop()

        selected_video_file = st.selectbox("Choose a video to search:", video_files)

        # Update the video path and clear stale playback/results on video changes.
        reset_search_session_for_video(
            st.session_state,
            selected_video_file,
            os.path.join(VIDEO_DATA_DIR, selected_video_file),
            os.path.splitext(selected_video_file)[0],
        )

        top_k = st.slider("Number of results", min_value=1, max_value=50, value=5)

    except FileNotFoundError:
        st.error(f"Video directory not found at '{VIDEO_DATA_DIR}'. Please create it and add videos.")
        st.stop()

# --- MAIN CONTENT AREA ---
# Create two columns: one for search results, one for the video player
col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("Search")
    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g., a man holding a gun",
        max_chars=1000,
    ).strip()
    
    if st.button("Search", type="primary"):
        if not query:
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching for relevant moments..."):
                try:
                    payload = search_payload(
                        query,
                        st.session_state.video_filename_clean,
                        top_k=top_k,
                    )
                    response = post_search(API_URL, payload)

                    if response.status_code == 200:
                        try:
                            st.session_state.search_results = search_results_from_response(
                                response
                            )
                            st.session_state.last_search_query = query
                        except ValueError as e:
                            st.error(f"Search API returned an unusable response: {e}")
                            st.session_state.search_results = []
                            st.session_state.last_search_query = None
                    else:
                        st.error(
                            f"Failed to get results from API (status {response.status_code}).\n\n"
                            f"Response: {response.text}"
                        )
                        st.session_state.search_results = []
                        st.session_state.last_search_query = None

                except RequestException as e:
                    st.error(
                        f"Could not connect to the Search API at {API_URL}. "
                        f"Is it running?\n\nDetails: {e}"
                    )
                    st.session_state.search_results = []
                    st.session_state.last_search_query = None

with col2:
    st.header("Player")
    if st.session_state.video_path:
        st.video(st.session_state.video_path, start_time=st.session_state.start_time)
    else:
        st.info("Select a video from the sidebar to begin.")

st.divider()

# --- DISPLAY SEARCH RESULTS ---
st.header("Results")
if st.session_state.search_results:
    results = st.session_state.search_results
    for result in results:
        with st.container(border=True):
            res_col1, res_col2 = st.columns([4, 1])
            with res_col1:
                st.subheader(f"**Title:** {result['title']}")
                st.write(f"**Summary:** {result['summary']}")
                time_range = format_time_range(result["start_time"], result["end_time"])
                st.caption(
                    f"**Time:** {time_range}  |  **Speakers:** "
                    f"{result['speakers'] or 'N/A'}"
                )
            with res_col2:
                # Use a unique key for each button to avoid conflicts
                if st.button("▶️ Play", key=f"play_{result['id']}"):
                    st.session_state.start_time = int(result['start_time'])
                    # st.rerun() is essential to force the UI to update immediately
                    # with the new video start time.
                    st.rerun()
elif st.session_state.last_search_query:
    st.info(f"No matches found for {st.session_state.last_search_query!r}.")
else:
    st.info("Your search results will appear here.")
