import logging
import json
import os
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol

from app.ui.speaker_support import normalize_speaker_map, speaker_ids_from_transcript

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
    "text": 0.35,      # Importance of the script/topic changing
    "visual": 0.20,    # Importance of the visuals changing (based on captions)
    "action": 0.15,    # Importance of the main activity changing
    "speaker": 0.15,   # Importance of the people speaking changing
    "audio_env": 0.10, # Importance of background sounds (music, applause) changing
    "silence": 0.05    # Importance of a long pause in dialogue
}

# How many seconds of silence between shots constitutes a significant "gap".
SILENCE_GAP_THRESHOLD_SEC = 2.0

# The model to use for generating text/visual embeddings.
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# =================================================================================


class EmbeddingModel(Protocol):
    def encode(self, sentences: List[str], **kwargs: Any) -> Any:
        ...


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_shot_time(shot: Dict[str, Any], shot_index: int, field_name: str) -> float:
    value = shot.get(field_name)
    if not _is_number(value):
        raise ValueError(f"shot at index {shot_index} must have numeric {field_name}")
    if value < 0:
        raise ValueError(f"shot at index {shot_index} must have non-negative {field_name}")
    return float(value)


def _validate_speaker_map(raw_speaker_map: Any) -> Dict[str, str]:
    speaker_map = normalize_speaker_map(raw_speaker_map)
    if speaker_map is None:
        raise ValueError(
            "speaker map must be a JSON object with non-empty string speaker IDs and names"
        )
    return speaker_map


def _validate_speaker_map_coverage(
    analysis_data: List[Dict[str, Any]],
    speaker_map: Dict[str, str],
) -> None:
    transcript_segments = [
        segment
        for shot in analysis_data
        for segment in shot.get("transcript_segments", [])
    ]
    missing_speaker_ids = sorted(
        set(speaker_ids_from_transcript(transcript_segments)) - set(speaker_map)
    )
    if missing_speaker_ids:
        raise ValueError(
            "speaker map is missing names for: " + ", ".join(missing_speaker_ids)
        )


def _validate_labeled_items(
    shot: Dict[str, Any],
    shot_index: int,
    field_name: str,
    label_name: str,
) -> None:
    items = shot.get(field_name, [])
    if not isinstance(items, list):
        raise ValueError(f"shot at index {shot_index} field {field_name} must be a JSON array")

    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"{field_name} item at shot index {shot_index}, index {item_index} "
                "must be a JSON object"
            )
        label = item.get(label_name)
        if not isinstance(label, str):
            raise ValueError(
                f"{field_name} item at shot index {shot_index}, index {item_index} "
                f"must have string {label_name}"
            )


def _validate_analysis_data(raw_analysis_data: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_analysis_data, list):
        raise ValueError("final analysis file must contain a JSON array")

    for shot_index, shot in enumerate(raw_analysis_data):
        if not isinstance(shot, dict):
            raise ValueError(f"shot at index {shot_index} must be a JSON object")

        shot_id = shot.get("shot_id")
        if not isinstance(shot_id, str) or not shot_id.strip():
            raise ValueError(f"shot at index {shot_index} must have a shot_id")

        time_start_sec = _validate_shot_time(shot, shot_index, "time_start_sec")
        time_end_sec = _validate_shot_time(shot, shot_index, "time_end_sec")
        if time_end_sec < time_start_sec:
            raise ValueError(
                f"shot at index {shot_index} "
                "time_end_sec must be greater than or equal to time_start_sec"
            )

        visual_caption = shot.get("visual_caption")
        if not isinstance(visual_caption, str):
            raise ValueError(f"shot at index {shot_index} must have string visual_caption")

        transcript_segments = shot.get("transcript_segments")
        if not isinstance(transcript_segments, list):
            raise ValueError(
                f"shot at index {shot_index} field transcript_segments must be a JSON array"
            )

        for segment_index, segment in enumerate(transcript_segments):
            if not isinstance(segment, dict):
                raise ValueError(
                    f"transcript segment at shot index {shot_index}, index {segment_index} "
                    "must be a JSON object"
                )
            if not isinstance(segment.get("text"), str):
                raise ValueError(
                    f"transcript segment at shot index {shot_index}, index {segment_index} "
                    "must have string text"
                )
            speaker = segment.get("speaker")
            if speaker is not None and not isinstance(speaker, str):
                raise ValueError(
                    f"transcript segment at shot index {shot_index}, index {segment_index} "
                    "must have string speaker"
                )

        _validate_labeled_items(shot, shot_index, "audio_events", "event")
        _validate_labeled_items(shot, shot_index, "detected_actions", "action")

    return raw_analysis_data


def _load_config() -> Dict[str, Any]:
    from core.config import CONFIG

    return CONFIG


def create_embedding_model(config: Dict[str, Any]) -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    device = config.get("general", {}).get("device", "cpu")
    embedding_model_name = (
        config.get("models", {})
        .get("embedding", {})
        .get("name", EMBEDDING_MODEL)
    )
    logger.info("   -> Loading sentence embedding model: '%s'...", embedding_model_name)
    return SentenceTransformer(embedding_model_name, device=device)


def _vector_to_floats(vector: Any) -> List[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _encoded_vectors_to_lists(
    encoded_vectors: Any,
    expected_count: int,
    embedding_type: str,
) -> List[List[float]]:
    if hasattr(encoded_vectors, "tolist"):
        encoded_vectors = encoded_vectors.tolist()

    try:
        vectors = [_vector_to_floats(vector) for vector in encoded_vectors]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{embedding_type} embeddings must be numeric vectors") from exc

    if len(vectors) != expected_count:
        raise ValueError(
            f"{embedding_type} embedding count {len(vectors)} "
            f"does not match shot count {expected_count}"
        )

    return vectors


def _cosine_similarity(left: Any, right: Any) -> float:
    left_values = _vector_to_floats(left)
    right_values = _vector_to_floats(right)

    numerator = sum(left_value * right_value for left_value, right_value in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return numerator / (left_norm * right_norm)


def run_segmentation(
    video_path: str,
    analysis_path: str,
    speaker_map_path: str,
    config: Optional[Dict[str, Any]] = None,
    embedding_model: Optional[EmbeddingModel] = None,
) -> Optional[str]:
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
    if config is None:
        config = _load_config()
    logger.info("--- Starting Step 2: Intelligent Segmentation with Boundary Scoring ---")

    # Define output path early from CONFIG to check for its existence
    processed_dir = os.path.dirname(analysis_path)
    output_filename = config.get('filenames', {}).get(
        'final_segments',
        'final_segments.json',
    )
    output_path = os.path.join(processed_dir, output_filename)

    # --- CHECK IF ALREADY COMPLETED ---
    if os.path.exists(output_path):
        logger.info(f"    -> Skipping segmentation. Output already exists at {output_path}")
        logger.info("--- Segmentation Step Complete (Skipped)! ---")
        return output_path
    
    # 1. Load all necessary data
    logger.info("1/4: Loading input data...")
    with open(analysis_path, 'r') as f:
        analysis_data = _validate_analysis_data(json.load(f))
    with open(speaker_map_path, 'r') as f:
        speaker_map = _validate_speaker_map(json.load(f))
    _validate_speaker_map_coverage(analysis_data, speaker_map)

    if not analysis_data:
        logger.warning("The 'final_analysis.json' file is empty. Cannot perform segmentation.")
        return None

    # 2. Pre-process and enrich shot data
    logger.info("2/4: Assembling and enriching shot data...")
    rich_shots = _prepare_rich_shots(analysis_data, speaker_map)

    # 3. Perform the segmentation algorithm
    logger.info("3/4: Performing boundary scoring segmentation...")
    if embedding_model is None:
        embedding_model = create_embedding_model(config)
    final_segments = _perform_boundary_scoring(rich_shots, embedding_model)

    # 4. Save the final segments to a file
    logger.info("4/4: Saving final segments...")
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
            generic_speaker = (seg.get('speaker') or 'UNKNOWN').strip()
            real_name = speaker_map.get(generic_speaker, "Unknown Speaker")
            shot_speakers.add(real_name)

        rich_shots.append({
            "shot_id": shot['shot_id'],
            "start_time": shot['time_start_sec'],
            "end_time": shot['time_end_sec'],
            "dialogue_text": dialogue_text,
            "visual_caption": shot['visual_caption'],
            "speakers": sorted(list(shot_speakers)),
            "audio_events": [event['event'] for event in shot.get('audio_events', [])],
            "actions": [action['action'] for action in shot.get('detected_actions', [])]
        })
    return rich_shots

def _perform_boundary_scoring(
    rich_shots: List[Dict[str, Any]],
    embedding_model: Optional[EmbeddingModel] = None,
) -> List[Dict[str, Any]]:
    """
    The core algorithm. Iterates through shots, calculates a Boundary Score between
    adjacent shots, and splits them into segments based on a threshold.
    """
    if not rich_shots:
        return []

    if embedding_model is None:
        embedding_model = create_embedding_model({})
    
    # Pre-calculate embeddings for all shots to avoid re-calculating in the loop.
    logger.info("   -> Pre-calculating text and visual embeddings for all shots...")
    dialogue_texts = [shot['dialogue_text'] for shot in rich_shots]
    visual_captions = [shot['visual_caption'] for shot in rich_shots]
    
    dialogue_embeddings = _encoded_vectors_to_lists(
        embedding_model.encode(
            dialogue_texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        ),
        len(rich_shots),
        "dialogue",
    )
    visual_embeddings = _encoded_vectors_to_lists(
        embedding_model.encode(
            visual_captions,
            show_progress_bar=True,
            normalize_embeddings=True,
        ),
        len(rich_shots),
        "visual",
    )

    final_segments = []
    current_segment_shots = [rich_shots[0]]

    for i in range(1, len(rich_shots)):
        prev_shot = rich_shots[i-1]
        current_shot = rich_shots[i]

        # --- 1. Textual Change Score ---
        # How much the topic of conversation has shifted.
        text_sim = _cosine_similarity(dialogue_embeddings[i-1], dialogue_embeddings[i])
        text_change = 1 - text_sim

        # --- 2. Visual Change Score ---
        # How much the visual content has shifted, using captions as a proxy.
        visual_sim = _cosine_similarity(visual_embeddings[i-1], visual_embeddings[i])
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

        # --- 5. Action Change Score ---
        # How much the core activity has changed.
        s1_action = set(prev_shot['actions'])
        s2_action = set(current_shot['actions'])
        intersection_len_action = len(s1_action.intersection(s2_action))
        union_len_action = len(s1_action.union(s2_action))
        jaccard_sim_action = intersection_len_action / union_len_action if union_len_action > 0 else 1
        action_change = 1 - jaccard_sim_action

        
        # --- 6. Silence Gap Score ---
        # A hard signal: 1 if there's a long pause between shots, 0 otherwise.
        time_gap = current_shot['start_time'] - prev_shot['end_time']
        silence_gap = 1 if time_gap > SILENCE_GAP_THRESHOLD_SEC else 0
        
        # --- Calculate the final weighted Boundary Score ---
        boundary_score = (
            (text_change * WEIGHTS["text"]) +
            (visual_change * WEIGHTS["visual"]) +
            (speaker_change * WEIGHTS["speaker"]) +
            (audio_env_change * WEIGHTS["audio_env"]) +
            (action_change * WEIGHTS["action"]) +
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
    all_actions = set()
    
    for shot in shots_to_merge:
        all_speakers.update(shot['speakers'])
        if shot['visual_caption'] not in all_visual_captions:
             all_visual_captions.append(shot['visual_caption'])
        all_audio_events.update(shot['audio_events'])
        all_actions.update(shot['actions']) 

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
        "consolidated_actions": sorted(list(all_actions)), 
        "shot_count": len(shots_to_merge),
        "shot_ids": [shot['shot_id'] for shot in shots_to_merge]
    }
