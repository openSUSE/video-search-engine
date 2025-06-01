import logging
import pandas as pd
import json
import os
import cv2 # OpenCV for getting video FPS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- GET A LOGGER FOR THIS MODULE ---
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# These are the "knobs" you can tune to change the segmentation behavior.
# Higher threshold = more likely to create a new segment.
BOUNDARY_SCORE_THRESHOLD = 0.5 
# Weights for the different change signals
WEIGHT_TEXT_CHANGE = 0.6
WEIGHT_SPEAKER_CHANGE = 0.3
WEIGHT_SILENCE_GAP = 0.1
# How many seconds of silence constitutes a "gap"
SILENCE_GAP_THRESHOLD = 2.0 

def run_segmentation(video_path: str, shots_path: str, transcript_path: str, speaker_map_path: str) -> str:
    """
    Performs intelligent segmentation on the video data.
    Returns the path to the final segments JSON file.
    """
    logger.info("--- Starting Step 2: Intelligent Segmentation ---")

    # 1. Load all necessary data
    logger.info("1/4: Loading input data...")
    shots_df = pd.read_csv(shots_path)
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    with open(speaker_map_path, 'r') as f:
        speaker_map = json.load(f)

    # Get video FPS using OpenCV
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    logger.info(f"   -> Video FPS: {fps}")

    # 2. Assemble the rich "micro-shot" data structure
    logger.info("2/4: Assembling micro-shots...")
    micro_shots = _assemble_micro_shots(shots_df, transcript_data, speaker_map, fps)
    
    # 3. Perform the segmentation algorithm
    logger.info("3/4: Performing boundary scoring segmentation...")
    final_segments = _perform_segmentation(micro_shots)

    # 4. Save the final segments to a file
    logger.info("4/4: Saving final segments...")
    processed_dir = "data/processed"
    output_path = os.path.join(processed_dir, "final_segments.json")
    with open(output_path, 'w') as f:
        json.dump(final_segments, f, indent=2)
    
    logger.info(f"--- Segmentation Step Complete! Saved to {output_path} ---")
    return output_path

def _assemble_micro_shots(shots_df, transcript_data, speaker_map, fps):
    """Combines shot boundaries and dialogue into a list of micro-shots."""
    micro_shots = []
    for index, row in shots_df.iterrows():
        start_time = row['start_frame'] / fps
        end_time = row['end_frame'] / fps
        
        dialogue_in_shot = []
        speakers_in_shot = set()
        
        # Find all dialogue that falls within this shot's timeframe
        for segment in transcript_data:
            seg_start = segment['start']
            seg_end = segment['end']
            if max(start_time, seg_start) < min(end_time, seg_end): # Check for overlap
                speaker_label = segment.get('speaker', 'UNKNOWN')
                speaker_name = speaker_map.get(speaker_label, 'Unknown Speaker')
                dialogue_in_shot.append({
                    "text": segment['text'].strip(),
                    "speaker": speaker_name
                })
                speakers_in_shot.add(speaker_name)

        micro_shots.append({
            "shot_id": index,
            "start_time": start_time,
            "end_time": end_time,
            "dialogue": dialogue_in_shot,
            "speakers": sorted(list(speakers_in_shot))
        })
    return micro_shots

def _perform_segmentation(micro_shots):
    """Runs the Boundary Scoring algorithm to group micro-shots."""
    if not micro_shots:
        return []

    # Load a sentence transformer model for text embeddings
    # The first time you run this, it will download the model.
    logger.info("   -> Loading sentence-transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # Pre-calculate embeddings for all shots
    logger.info("   -> Pre-calculating text embeddings for all shots...")
    shot_texts = [" ".join(d['text'] for d in shot['dialogue']) for shot in micro_shots]
    shot_embeddings = model.encode(shot_texts, show_progress_bar=True)

    final_segments = []
    current_segment_shots = [micro_shots[0]]

    for i in range(1, len(micro_shots)):
        prev_shot = micro_shots[i-1]
        current_shot = micro_shots[i]

        # Calculate the "change signals"
        # 1. Textual Change
        prev_embedding = shot_embeddings[i-1].reshape(1, -1)
        current_embedding = shot_embeddings[i].reshape(1, -1)
        text_similarity = cosine_similarity(prev_embedding, current_embedding)[0][0]
        textual_change_score = 1 - text_similarity
        
        # 2. Speaker Change
        prev_speakers = set(prev_shot['speakers'])
        current_speakers = set(current_shot['speakers'])
        speaker_change_score = 0 if prev_speakers == current_speakers else 1

        # 3. Silence Gap
        time_gap = current_shot['start_time'] - prev_shot['end_time']
        silence_gap_score = 1 if time_gap > SILENCE_GAP_THRESHOLD else 0
        
        # Calculate the final Boundary Score
        boundary_score = (
            (textual_change_score * WEIGHT_TEXT_CHANGE) +
            (speaker_change_score * WEIGHT_SPEAKER_CHANGE) +
            (silence_gap_score * WEIGHT_SILENCE_GAP)
        )

        # Decision: To merge or to split?
        if boundary_score > BOUNDARY_SCORE_THRESHOLD:
            # Finalize the previous segment
            final_segments.append(_merge_shots_into_segment(current_segment_shots))
            # Start a new segment
            current_segment_shots = [current_shot]
        else:
            # Merge the current shot into the ongoing segment
            current_segment_shots.append(current_shot)

    # Don't forget to add the very last segment after the loop finishes
    if current_segment_shots:
        final_segments.append(_merge_shots_into_segment(current_segment_shots))

    return final_segments

def _merge_shots_into_segment(shots_to_merge):
    """Helper function to combine a list of shots into a single segment object."""
    if not shots_to_merge:
        return {}
    
    full_transcript = " ".join(
        d['text'] for shot in shots_to_merge for d in shot['dialogue']
    ).strip()
    
    all_speakers = set()
    for shot in shots_to_merge:
        all_speakers.update(shot['speakers'])

    return {
        "start_time": shots_to_merge[0]['start_time'],
        "end_time": shots_to_merge[-1]['end_time'],
        "full_transcript": full_transcript,
        "speakers": sorted(list(all_speakers))
    }
