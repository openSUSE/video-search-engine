import streamlit as st
import os
import json
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Speaker Identification Tool")
st.title("🗣️ Manual Speaker Identification Tool")

# --- INITIALIZE SESSION STATE ---
# Session state is used to store variables that persist across reruns of the script.
if 'speaker_map' not in st.session_state:
    st.session_state.speaker_map = {}
if 'video_start_time' not in st.session_state:
    st.session_state.video_start_time = 0
if 'current_transcript_data' not in st.session_state:
    st.session_state.current_transcript_data = None


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
base_dir = st.text_input("Enter the base path for processed data:", "data/processed")

if not os.path.isdir(base_dir):
    st.error("The provided directory does not exist.")
else:
    # Get a list of all video subdirectories
    video_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not video_folders:
        st.warning("No processed video folders found in the specified directory.")
    else:
        selected_video_folder = st.selectbox("Select a video to process:", video_folders)
        video_specific_dir = os.path.join(base_dir, selected_video_folder)
        
        # Define paths for the required files
        transcript_path = os.path.join(video_specific_dir, "transcript_generic.json")
        video_path_original = os.path.join("data/videos", f"{selected_video_folder}.mp4") # Assuming this structure
        speaker_map_path = os.path.join(video_specific_dir, "speaker_map.json")

        # --- MAIN LOGIC ---
        if not os.path.exists(transcript_path):
            st.error(f"Transcript file not found: {transcript_path}")
        elif not os.path.exists(video_path_original):
            st.error(f"Original video file not found: {video_path_original}")
        else:
            # Load transcript data
            with open(transcript_path, 'r') as f:
                st.session_state.current_transcript_data = json.load(f)
            
            # Load existing speaker map if it exists
            if os.path.exists(speaker_map_path):
                with open(speaker_map_path, 'r') as f:
                    st.session_state.speaker_map = json.load(f)

            # Find all unique speaker labels from the transcript
            all_speakers = sorted(list(set(
                seg['speaker'] for seg in st.session_state.current_transcript_data if 'speaker' in seg
            )))
            
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
                    st.success("All speakers have been identified! The speaker map has been saved.")
                else:
                    st.subheader("❓ Unidentified Speakers")
                    selected_speaker_label = st.radio("Select a speaker to identify:", unidentified_speakers)
                    
                    speaker_name_input = st.text_input(f"Enter name for {selected_speaker_label}:", key=f"name_{selected_speaker_label}")
                    
                    if st.button(f"Assign Name to {selected_speaker_label}"):
                        if speaker_name_input:
                            st.session_state.speaker_map[selected_speaker_label] = speaker_name_input
                            # Save the map immediately on change
                            with open(speaker_map_path, 'w') as f:
                                json.dump(st.session_state.speaker_map, f, indent=2)
                            st.experimental_rerun() # Rerun the app to update the UI
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
                st.video(video_path_original, start_time=st.session_state.video_start_time)

