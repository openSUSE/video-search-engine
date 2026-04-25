import json
import os

import streamlit as st

from app.ui.path_settings import env_path_setting
from app.ui.speaker_support import (
    ensure_speaker_session_state,
    load_transcript_segments,
    load_speaker_map,
    normalize_speaker_map,
    processed_video_folders,
    reset_speaker_session_for_video,
    save_speaker_map_if_complete,
    speaker_artifact_paths,
    speaker_ids_from_transcript,
)

OUTPUT_DIR = env_path_setting("OUTPUT_DIR", "data/processed")
VIDEO_DATA_DIR = env_path_setting("VIDEO_DATA_PATH", "data/videos")


def get_config():
    from core.config import CONFIG

    return CONFIG

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Speaker Identification Tool")
st.title("🗣️ Manual Speaker Identification Tool")

# --- INITIALIZE SESSION STATE ---
# Session state is used to store variables that persist across reruns of the script.
ensure_speaker_session_state(st.session_state)


# --- UI LAYOUT ---
st.sidebar.header("Instructions")
st.sidebar.info(
    "1. Select the base directory where your processed video folders are located.\n"
    "2. Choose a specific video to work on.\n"
    "3. For each unidentified speaker, click on their dialogue segments to play them in the video.\n"
    "4. Enter the correct name and click 'Assign Name'.\n"
    "5. Once all speakers are identified, the map will be saved automatically."
)

# --- DIRECTORY AND VIDEO SELECTION ---
base_dir = st.text_input("Enter the base path for processed data:", OUTPUT_DIR)

if not os.path.isdir(base_dir):
    st.error("The provided directory does not exist.")
else:
    # Get a list of all video subdirectories
    video_folders = processed_video_folders(base_dir)
    
    if not video_folders:
        st.warning("No processed video folders found in the specified directory.")
    else:
        selected_video_folder = st.selectbox("Select a video to process:", video_folders)
        reset_speaker_session_for_video(st.session_state, selected_video_folder, base_dir)

        # Define paths for the required files
        paths = speaker_artifact_paths(
            base_dir,
            selected_video_folder,
            VIDEO_DATA_DIR,
            get_config(),
        )

        # --- MAIN LOGIC ---
        if not paths.transcript.exists():
            st.error(f"Transcript file not found: {paths.transcript}")
        elif not paths.video.exists():
            st.error(f"Original video file not found: {paths.video}")
        else:
            # Load transcript data
            try:
                st.session_state.current_transcript_data = load_transcript_segments(
                    paths.transcript
                )
            except (json.JSONDecodeError, ValueError) as exc:
                st.error(f"Transcript file is not usable: {exc}")
                st.stop()
            
            # Load existing speaker map if it exists
            if paths.speaker_map.exists():
                st.session_state.speaker_map = load_speaker_map(paths.speaker_map)

            # Find all unique speaker labels from the transcript
            all_speakers = speaker_ids_from_transcript(st.session_state.current_transcript_data)
            
            # Filter out speakers that have already been identified
            unidentified_speakers = [s for s in all_speakers if s not in st.session_state.speaker_map]

            # --- DISPLAY AND INTERACTION ---
            left_col, right_col = st.columns([1, 1.5]) # Adjust column ratio

            with left_col:
                st.header("Speaker Assignment")

                if st.session_state.speaker_map:
                    st.subheader("✅ Identified Speakers")
                    st.json(st.session_state.speaker_map)

                if not unidentified_speakers:
                    if save_speaker_map_if_complete(
                        paths.speaker_map,
                        st.session_state.speaker_map,
                        all_speakers,
                    ):
                        st.success("All speakers have been identified! The speaker map has been saved.")
                    else:
                        st.warning("Speaker assignments are incomplete.")
                else:
                    st.subheader("❓ Unidentified Speakers")
                    selected_speaker_label = st.radio("Select a speaker to identify:", unidentified_speakers)
                    
                    speaker_name_input = st.text_input(f"Enter name for {selected_speaker_label}:", key=f"name_{selected_speaker_label}")
                    
                    if st.button(f"Assign Name to {selected_speaker_label}"):
                        updated_speaker_map = {
                            **st.session_state.speaker_map,
                            selected_speaker_label: speaker_name_input,
                        }
                        normalized_speaker_map = normalize_speaker_map(updated_speaker_map)
                        if normalized_speaker_map:
                            st.session_state.speaker_map = normalized_speaker_map
                            # Save the map immediately on change
                            with paths.speaker_map.open("w") as f:
                                json.dump(st.session_state.speaker_map, f, indent=2)
                            st.rerun() # Rerun the app to update the UI
                        else:
                            st.warning("Please enter a name.")

                    st.subheader("Dialogue Segments")
                    st.write("Click a segment to play it in the video.")
                    
                    # Display clickable dialogue segments for the selected speaker
                    for segment in st.session_state.current_transcript_data:
                        if segment.get('speaker') == selected_speaker_label:
                            start_time = segment['start']
                            text = segment['text']
                            if st.button(f"[{start_time:.1f}s] {text[:70]}...", key=f"seg_{start_time}"):
                                st.session_state.video_start_time = int(start_time)

            with right_col:
                st.header("Video Player")
                st.video(str(paths.video), start_time=st.session_state.video_start_time)
