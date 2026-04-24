# app/ui/search_app.py

import os
import sys

import streamlit as st

VIDEO_DATA_DIR = os.getenv("VIDEO_DATA_PATH", "data/videos")

# Add the project root to the Python path to allow importing from 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.ui.search_client import post_search, search_api_url, search_payload
from core.config import CONFIG

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Semantic Video Search",
    layout="wide"
)

# --- LOAD CONFIGURATION ---
try:
    API_URL = search_api_url(CONFIG)
except Exception as e:
    st.error(f"Error accessing configuration. Details: {e}")
    st.stop()

# --- SESSION STATE INITIALIZATION ---
# Session state holds variables that persist across user interactions.
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0

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
        
        # Update the video path in the session state when a new video is selected
        st.session_state.video_path = os.path.join(VIDEO_DATA_DIR, selected_video_file)
        st.session_state.video_filename_clean = os.path.splitext(selected_video_file)[0]

    except FileNotFoundError:
        st.error(f"Video directory not found at '{VIDEO_DATA_DIR}'. Please create it and add videos.")
        st.stop()

# --- MAIN CONTENT AREA ---
# Create two columns: one for search results, one for the video player
col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("Search")
    query = st.text_input("What are you looking for?", placeholder="e.g., a man holding a gun")
    
    if st.button("Search", type="primary"):
        if not query:
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching for relevant moments..."):
                try:
                    payload = search_payload(
                        query,
                        st.session_state.video_filename_clean,
                        top_k=5,
                    )
                    response = post_search(API_URL, payload)
                    
                    if response.status_code == 200:
                        results = response.json().get('results', [])
                        st.session_state.search_results = results
                    else:
                        st.error(f"Failed to get results from API. Status code: {response.status_code}")
                        st.error(f"Response: {response.text}")
                        st.session_state.search_results = []

                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to the Search API at {API_URL}. Is it running?")
                    st.error(f"Details: {e}")
                    st.session_state.search_results = []

with col2:
    st.header("Player")
    if st.session_state.video_path:
        st.video(st.session_state.video_path, start_time=st.session_state.start_time)
    else:
        st.info("Select a video from the sidebar to begin.")

st.divider()

# --- DISPLAY SEARCH RESULTS ---
st.header("Results")
if 'search_results' in st.session_state and st.session_state.search_results:
    results = st.session_state.search_results
    for result in results:
        with st.container(border=True):
            res_col1, res_col2 = st.columns([4, 1])
            with res_col1:
                st.subheader(f"**Title:** {result['title']}")
                st.write(f"**Summary:** {result['summary']}")
                st.caption(f"**Time:** {int(result['start_time']//60)}m {int(result['start_time']%60)}s  |  **Speakers:** {result['speakers'] or 'N/A'}")
            with res_col2:
                # Use a unique key for each button to avoid conflicts
                if st.button("▶️ Play", key=f"play_{result['id']}"):
                    st.session_state.start_time = int(result['start_time'])
                    # st.rerun() is essential to force the UI to update immediately
                    # with the new video start time.
                    st.rerun()
else:
    st.info("Your search results will appear here.")
