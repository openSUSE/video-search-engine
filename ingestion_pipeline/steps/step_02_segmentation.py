import logging
import json
import os
import cv2  # OpenCV for getting video FPS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any, Set
from collections import defaultdict
from core.config import CONFIG  # Import the global config object

# --- GET A LOGGER FOR THIS MODULE ---
logger = logging.getLogger(__name__)

# =================================================================================
# --- CONFIGURATION: These are the "knobs" to tune segmentation behavior ---
# =================================================================================

# The overall threshold for creating a new segment. Higher = fewer, longer segments.
BOUNDARY_SCORE_THRESHOLD = 0.45

# Weights for the different "change signals". These MUST sum to 1.0.
# Tune these to prioritize different types of changes.
WEIGHTS = {
    "text": 0.40,      # Importance of the script/topic changing
    "visual": 0.25,    # Importance of the visuals changing (based on captions)
    "speaker": 0.20,   # Importance of the people speaking changing
    "audio_env": 0.10, # Importance of background sounds (music, applause) changing
    "silence": 0.05    # Importance of a long pause in dialogue
}

# How many seconds of silence between shots constitutes a significant "gap".
SILENCE_GAP_THRESHOLD_SEC = 2.0

# The model to use for generating text/visual embeddings.
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# =================================================================================

def run_segmentation(video_path: str, analysis_path: str, speaker_map_path: str) -> str:
    """
    Performs intelligent segmentation by grouping shots based on a composite "Boundary Score".
    This function implements the "Phase 2: Intelligent Segmentation" logic.

    Args:
        video_path: Path to the original video file (for FPS).
        analysis_path: Path to the 'final_analysis.json' file from the extraction step.
        speaker_map_path: Path to the JSON file mapping generic speaker IDs to real names.

    Returns:
        The path to the final segments JSON file.
    """
    logger.info("--- Starting Step 2: Intelligent Segmentation with Boundary Scoring ---")

    # Define output path early from CONFIG to check for its existence
    processed_dir = os.path.dirname(analysis_path)
    output_filename = CONFIG['filenames']['final_segments']
    output_path = os.path.join(processed_dir, output_filename)

    # --- CHECK IF ALREADY COMPLETED ---
    if os.path.exists(output_path):
        logger.info(f"    -> Skipping segmentation. Output already exists at {output_path}")
        logger.info("--- Segmentation Step Complete (Skipped)! ---")
        return output_path
    
    # 1. Load all necessary data
    logger.info("1/4: Loading input data...")
    with open(analysis_path, 'r') as f:
        analysis_data = json.load(f)
    with open(speaker_map_path, 'r') as f:
        speaker_map = json.load(f)

    if not analysis_data:
        logger.warning("The 'final_analysis.json' file is empty. Cannot perform segmentation.")
        return None

    # 2. Pre-process and enrich shot data
    logger.info("2/4: Assembling and enriching shot data...")
    rich_shots = _prepare_rich_shots(analysis_data, speaker_map)

    # 3. Perform the segmentation algorithm
    logger.info("3/4: Performing boundary scoring segmentation...")
    final_segments = _perform_boundary_scoring(rich_shots)

    # 4. Save the final segments to a file
    logger.info("4/4: Saving final segments...")
    processed_dir = os.path.dirname(analysis_path) # Save in the same directory
    output_path = os.path.join(processed_dir, "final_segments.json")
    with open(output_path, 'w') as f:
        json.dump(final_segments, f, indent=4)
    
    logger.info(f"--- Segmentation Step Complete! Saved {len(final_segments)} segments to {output_path} ---")
    return output_path

def _prepare_rich_shots(analysis_data: List[Dict[str, Any]], speaker_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """Enriches the raw analysis data for each shot with speaker names and consolidated text."""
    rich_shots = []
    for shot in analysis_data:
        # Consolidate dialogue and map speaker names
        dialogue_text = " ".join(seg['text'].strip() for seg in shot['transcript_segments'])
        
        # Get unique, real speaker names for the shot
        shot_speakers = set()
        for seg in shot['transcript_segments']:
            generic_speaker = seg.get('speaker', 'UNKNOWN')
            real_name = speaker_map.get(generic_speaker, "Unknown Speaker")
            shot_speakers.add(real_name)

        rich_shots.append({
            "shot_id": shot['shot_id'],
            "start_time": shot['time_start_sec'],
            "end_time": shot['time_end_sec'],
            "dialogue_text": dialogue_text,
            "visual_caption": shot['visual_caption'],
            "speakers": sorted(list(shot_speakers)),
            "audio_events": [event['event'] for event in shot.get('audio_events', [])]
        })
    return rich_shots

def _perform_boundary_scoring(rich_shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    The core algorithm. Iterates through shots, calculates a Boundary Score between
    adjacent shots, and splits them into segments based on a threshold.
    """
    if not rich_shots:
        return []

    # Load a sentence transformer model. This will download the model on the first run.
    logger.info(f"   -> Loading sentence embedding model: '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
    
    # Pre-calculate embeddings for all shots to avoid re-calculating in the loop.
    logger.info("   -> Pre-calculating text and visual embeddings for all shots...")
    dialogue_texts = [shot['dialogue_text'] for shot in rich_shots]
    visual_captions = [shot['visual_caption'] for shot in rich_shots]
    
    dialogue_embeddings = model.encode(dialogue_texts, show_progress_bar=True, normalize_embeddings=True)
    visual_embeddings = model.encode(visual_captions, show_progress_bar=True, normalize_embeddings=True)

    final_segments = []
    current_segment_shots = [rich_shots[0]]

    for i in range(1, len(rich_shots)):
        prev_shot = rich_shots[i-1]
        current_shot = rich_shots[i]

        # --- 1. Textual Change Score ---
        # How much the topic of conversation has shifted.
        text_sim = cosine_similarity(dialogue_embeddings[i-1].reshape(1, -1), dialogue_embeddings[i].reshape(1, -1))[0][0]
        text_change = 1 - text_sim

        # --- 2. Visual Change Score ---
        # How much the visual content has shifted, using captions as a proxy.
        visual_sim = cosine_similarity(visual_embeddings[i-1].reshape(1, -1), visual_embeddings[i].reshape(1, -1))[0][0]
        visual_change = 1 - visual_sim

        # --- 3. Speaker Change Score ---
        # A hard signal: 1 if the set of speakers changes, 0 otherwise.
        speaker_change = 0 if set(prev_shot['speakers']) == set(current_shot['speakers']) else 1

        # --- 4. Audio Environment Change Score ---
        # How much the background soundscape has changed (e.g., from talking to music).
        # We use Jaccard distance (1 - Jaccard similarity) on the sets of audio events.
        s1 = set(prev_shot['audio_events'])
        s2 = set(current_shot['audio_events'])
        intersection_len = len(s1.intersection(s2))
        union_len = len(s1.union(s2))
        jaccard_sim = intersection_len / union_len if union_len > 0 else 1
        audio_env_change = 1 - jaccard_sim
        
        # --- 5. Silence Gap Score ---
        # A hard signal: 1 if there's a long pause between shots, 0 otherwise.
        time_gap = current_shot['start_time'] - prev_shot['end_time']
        silence_gap = 1 if time_gap > SILENCE_GAP_THRESHOLD_SEC else 0
        
        # --- Calculate the final weighted Boundary Score ---
        boundary_score = (
            (text_change * WEIGHTS["text"]) +
            (visual_change * WEIGHTS["visual"]) +
            (speaker_change * WEIGHTS["speaker"]) +
            (audio_env_change * WEIGHTS["audio_env"]) +
            (silence_gap * WEIGHTS["silence"])
        )

        # --- Decision: To merge or to split? ---
        if boundary_score > BOUNDARY_SCORE_THRESHOLD:
            # The change is significant. Finalize the previous segment.
            final_segments.append(_merge_shots_into_segment(current_segment_shots))
            # Start a new segment with the current shot.
            current_segment_shots = [current_shot]
        else:
            # The change is not significant. Merge the current shot into the ongoing segment.
            current_segment_shots.append(current_shot)

    # Don't forget to add the very last segment after the loop finishes.
    if current_segment_shots:
        final_segments.append(_merge_shots_into_segment(current_segment_shots))

    # Re-index the segments for clean output
    for i, segment in enumerate(final_segments):
        segment['segment_id'] = f"segment_{i+1:04d}"
        segment['segment_index'] = i + 1

    return final_segments

def _merge_shots_into_segment(shots_to_merge: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Helper function to combine a list of rich shots into a single segment object."""
    if not shots_to_merge:
        return {}
    
    # Consolidate all information from the merged shots
    full_transcript = " ".join(shot['dialogue_text'] for shot in shots_to_merge if shot['dialogue_text']).strip()
    
    all_speakers = set()
    all_visual_captions = []
    all_audio_events = set()
    
    for shot in shots_to_merge:
        all_speakers.update(shot['speakers'])
        if shot['visual_caption'] not in all_visual_captions:
             all_visual_captions.append(shot['visual_caption'])
        all_audio_events.update(shot['audio_events'])

    return {
        "segment_id": None, # Will be added later
        "segment_index": None, # Will be added later
        "start_time": shots_to_merge[0]['start_time'],
        "end_time": shots_to_merge[-1]['end_time'],
        "duration_sec": round(shots_to_merge[-1]['end_time'] - shots_to_merge[0]['start_time'], 3),
        "speakers": sorted(list(all_speakers)),
        "full_transcript": full_transcript,
        "consolidated_visual_captions": all_visual_captions,
        "consolidated_audio_events": sorted(list(all_audio_events)),
        "shot_count": len(shots_to_merge),
        "shot_ids": [shot['shot_id'] for shot in shots_to_merge]
    }
