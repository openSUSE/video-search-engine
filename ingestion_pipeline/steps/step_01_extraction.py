import subprocess
import whisperx
import torch
import logging
import os
import json
import csv
import cv2
from PIL import Image
from typing import Dict, Any, List
from collections import defaultdict
import numpy as np
from core.config import CONFIG

from ingestion_pipeline.utils.metadata_fetcher import fetch_movie_metadata

logger = logging.getLogger(__name__)

# Added path for the final unified output file.
def _get_paths(processed_dir: str, config: Dict[str, Any]) -> dict:
    """Generates a dictionary of all required output paths using filenames from config."""
    f_names = config['filenames']
    return {
        "audio": os.path.join(processed_dir, f_names['audio']),
        "shots": os.path.join(processed_dir, f_names['shots']),
        "transcript_raw": os.path.join(processed_dir, 'transcript_raw.json'),
        "transcript_aligned": os.path.join(processed_dir, f_names['transcript']),
        "audio_events": os.path.join(processed_dir, f_names['audio_events']),
        "visual_details": os.path.join(processed_dir, f_names['visual_details']),
        "actions": os.path.join(processed_dir, f_names['actions']),
        "final_analysis": os.path.join(processed_dir, f_names['final_analysis']),
    }

def extract_audio(video_path: str, audio_path: str):
    """Extracts and normalizes audio from a video file."""
    logger.info("    -> Extracting and normalizing audio...")
    command = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'libmp3lame', audio_path]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"    -> Audio saved to {audio_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed to process the audio.\nFFmpeg stderr:\n{e.stderr}")
        raise

def transcribe_and_diarize(audio_path: str, raw_transcript_path: str, config: Dict[str, Any]):
    """Transcribes audio and performs speaker diarization, saving the raw output."""
    logger.info("    -> Transcribing and identifying speakers (raw output)...")
    device = config['general']['device']
    model_cfg = config['models']['transcription']
    params_cfg = config['parameters']['transcription']
    
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model(model_cfg['name'], device, compute_type=model_cfg['compute_type'])
    result_transcript = model.transcribe(audio, batch_size=params_cfg['batch_size'])

    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=config['general']['hf_token'], device=device)
    diarize_segments = diarize_model(audio)
    result_transcript = whisperx.assign_word_speakers(diarize_segments, result_transcript)

    with open(raw_transcript_path, 'w') as f:
        json.dump(result_transcript['segments'], f, indent=2)
    logger.info(f"    -> Raw transcript saved to {raw_transcript_path}")

def detect_shot_boundaries(video_path: str, shots_path: str) -> List[Dict[str, Any]]:
    """Detects shot boundaries and saves them as a rich JSON object."""
    logger.info("    -> Detecting shot boundaries with TransNetV2...")
    from transnetv2_pytorch import TransNetV2
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    model_transnet = TransNetV2()
    _, _, all_frame_predictions = model_transnet.predict_video(video_path)
    scenes_frames = model_transnet.predictions_to_scenes(all_frame_predictions.cpu().numpy()).tolist()
    
    scenes_data = []
    for i, (start_frame, end_frame) in enumerate(scenes_frames):
        scenes_data.append({
            "shot_id": f"shot_{i+1:04d}",
            "shot_index": i + 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_sec": round(start_frame / fps, 3),
            "end_time_sec": round(end_frame / fps, 3)
        })

    with open(shots_path, 'w') as f:
        json.dump(scenes_data, f, indent=2)
    logger.info(f"    -> Shot boundaries saved to {shots_path}")
    return scenes_data

def align_transcript_to_shots(raw_transcript_path: str, scenes: List[Dict[str, Any]], aligned_transcript_path: str):
    """Aligns transcript segments to shots and saves the new transcript."""
    logger.info("    -> Aligning transcript to shots...")
    with open(raw_transcript_path, 'r') as f:
        transcript_segments = json.load(f)

    aligned_segments = []
    for segment in transcript_segments:
        if 'start' not in segment or 'end' not in segment: continue
        segment_midpoint = (segment['start'] + segment['end']) / 2
        assigned_shot_id = None
        for shot in scenes:
            if shot['start_time_sec'] <= segment_midpoint < shot['end_time_sec']:
                assigned_shot_id = shot['shot_id']
                break
        
        aligned_segment = {
            "start": segment.get('start'), "end": segment.get('end'),
            "text": segment.get('text', ''), "speaker": segment.get('speaker'),
            "shot_id": assigned_shot_id
        }
        aligned_segments.append(aligned_segment)

    with open(aligned_transcript_path, 'w') as f:
        json.dump(aligned_segments, f, indent=2)
    logger.info(f"    -> Aligned transcript saved to {aligned_transcript_path}")

def detect_audio_events_per_shot(audio_path: str, scenes: List[Dict[str, Any]], output_path: str, config: Dict[str, Any]):
    """Detects audio events for each shot."""
    logger.info("    -> Detecting audio events per shot...")
    device = config['general']['device']
    model_cfg = config['models']['audio_events']
    audio_params = config['parameters']['audio']
    event_params = config['parameters']['audio_events']

    from transformers import AutoProcessor, AutoModelForAudioClassification
    import librosa

    processor = AutoProcessor.from_pretrained(model_cfg['name'])
    model = AutoModelForAudioClassification.from_pretrained(model_cfg['name']).to(device)
    sr = audio_params['sample_rate']
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    all_shot_events = []
    for shot in scenes:
        start_time, end_time = shot['start_time_sec'], shot['end_time_sec']
        audio_chunk = y[int(start_time * sr):int(end_time * sr)]
        shot_events_info = {"shot_id": shot["shot_id"], "events": []}

        if audio_chunk.shape[0] > 0:
            inputs = processor(audio_chunk, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            
            scores = torch.sigmoid(logits[0]).cpu().numpy()
            top_indices = scores.argsort()[-event_params['top_n']:][::-1]
            detected_events = [{"event": model.config.id2label[j], "score": round(float(scores[j]), 3)} 
                               for j in top_indices if scores[j] > event_params['confidence_threshold']]
            shot_events_info["events"] = detected_events
        all_shot_events.append(shot_events_info)

    with open(output_path, 'w') as f:
        json.dump(all_shot_events, f, indent=2)
    logger.info(f"    -> Timestamped audio events saved.")

def generate_visual_captions(video_path: str, scenes: List[Dict[str, Any]], output_path: str, config: Dict[str, Any]):
    """Generates captions for each shot."""
    logger.info("    -> Generating visual captions for shots...")
    device = config['general']['device']
    model_cfg = config['models']['visual_captioning']
    params_cfg = config['parameters']['visual_captioning']

    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained(model_cfg['name'])
    model = BlipForConditionalGeneration.from_pretrained(model_cfg['name']).to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"Cannot open video file: {video_path}")
    visual_details = []
    for shot in scenes:
        middle_frame_idx = (shot['start_frame'] + shot['end_frame']) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()

        if ret:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(pil_image, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=params_cfg['max_new_tokens'])
            caption = processor.decode(out[0], skip_special_tokens=True)
            visual_details.append({"shot_id": shot["shot_id"], "caption": caption})
    cap.release()

    with open(output_path, 'w') as f:
        json.dump(visual_details, f, indent=2)
    logger.info(f"    -> Visual details saved.")

def detect_actions_per_shot(video_path: str, scenes: List[Dict[str, Any]], output_path: str, config: Dict[str, Any]):
    """
    Detects actions and activities for each shot using a video classification model.
    """
    logger.info("    -> Detecting actions/activities per shot...")
    device = config['general']['device']
    model_cfg = config['models']['action_recognition']
    params_cfg = config['parameters']['action_recognition']
    num_frames_to_sample = params_cfg['num_frames']

    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

    processor = VideoMAEImageProcessor.from_pretrained(model_cfg['name'])
    model = VideoMAEForVideoClassification.from_pretrained(model_cfg['name']).to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    all_shot_actions = []
    for shot in scenes:
        start_frame, end_frame = shot['start_frame'], shot['end_frame']
        
        # Ensure the shot is long enough to sample from
        if end_frame - start_frame < num_frames_to_sample:
            all_shot_actions.append({"shot_id": shot["shot_id"], "actions": []})
            continue

        # Generate evenly spaced frame indices to sample from the shot
        frame_indices = np.linspace(start_frame, end_frame, num_frames_to_sample, dtype=int)
        
        shot_frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert frame from BGR (OpenCV) to RGB (transformers)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                shot_frames.append(rgb_frame)

        shot_actions_info = {"shot_id": shot["shot_id"], "actions": []}
        if shot_frames:
            # Process the collected frames and perform inference
            inputs = processor(shot_frames, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            # Get top N predictions
            scores = torch.softmax(logits, dim=-1)[0]
            top_predictions = torch.topk(scores, k=params_cfg['top_n'])
            
            detected_actions = []
            for i in range(params_cfg['top_n']):
                score = top_predictions.values[i].item()
                label_id = top_predictions.indices[i].item()
                action = model.config.id2label[label_id]
                detected_actions.append({"action": action, "score": round(score, 3)})
            
            shot_actions_info["actions"] = detected_actions
        
        all_shot_actions.append(shot_actions_info)

    cap.release()
    
    with open(output_path, 'w') as f:
        json.dump(all_shot_actions, f, indent=2)
    logger.info(f"    -> Detected actions saved to {output_path}")


# NEW: Function to combine all metadata into a single file.
def create_final_analysis_file(paths: Dict[str, str]):
    """Combines all intermediate JSON files into a single, unified analysis file."""
    logger.info("    -> Creating final unified analysis file...")

    # Load all the data sources
    with open(paths['shots'], 'r') as f: scenes_data = json.load(f)
    with open(paths['visual_details'], 'r') as f: visual_data = json.load(f)
    with open(paths['audio_events'], 'r') as f: audio_data = json.load(f)
    with open(paths['transcript_aligned'], 'r') as f: transcript_data = json.load(f)
    with open(paths['actions'], 'r') as f: actions_data = json.load(f)

    # Create maps for efficient lookup by shot_id
    captions_map = {item['shot_id']: item['caption'] for item in visual_data}
    audio_events_map = {item['shot_id']: item['events'] for item in audio_data}
    actions_map = {item['shot_id']: item['actions'] for item in actions_data} 
    
    # Group transcript segments by shot_id
    transcript_map = defaultdict(list)
    for segment in transcript_data:
        if segment['shot_id']:
            transcript_map[segment['shot_id']].append({
                "start": segment["start"], "end": segment["end"],
                "text": segment["text"], "speaker": segment["speaker"]
            })

    # Build the final combined data structure
    final_data = []
    for shot in scenes_data:
        shot_id = shot['shot_id']
        final_shot_object = {
            "shot_id": shot_id,
            "shot_index": shot['shot_index'],
            "time_start_sec": shot['start_time_sec'],
            "time_end_sec": shot['end_time_sec'],
            "frame_start": shot['start_frame'],
            "frame_end": shot['end_frame'],
            "visual_caption": captions_map.get(shot_id, ""),
            "detected_actions": actions_map.get(shot_id, []),
            "audio_events": audio_events_map.get(shot_id, []),
            "transcript_segments": transcript_map.get(shot_id, [])
        }
        final_data.append(final_shot_object)

    with open(paths['final_analysis'], 'w') as f:
        json.dump(final_data, f, indent=2)
    logger.info(f"    -> Final analysis file saved to {paths['final_analysis']}")

# --- MAIN ORCHESTRATOR FUNCTION ---
def run_extraction(video_path: str, base_output_dir: str, video_title: str = None, video_year: int = None):
    """Runs the full data extraction pipeline for a given video."""
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    video_specific_dir = os.path.join(base_output_dir, video_filename)
    
    logger.info(f"--- Starting Step 1: Data Extraction for '{video_filename}' ---")
    logger.info(f"Output will be saved in: {video_specific_dir}")
    
    os.makedirs(video_specific_dir, exist_ok=True)
    paths = _get_paths(video_specific_dir, CONFIG)

    logger.info(f"Attempting to fetch metadata for '{video_title}'...")
    fetched_metadata = fetch_movie_metadata(video_title, video_year)

    video_metadata = {
        "title": video_title or video_filename,
        "logline": "No logline provided.",
        "genre": "N/A",
        "setting": "N/A",
        "main_characters": []
    }

    if video_title:
        logger.info(f"Attempting to fetch metadata for '{video_title}'...")
        fetched_metadata = fetch_movie_metadata(video_title, video_year)
        
        if fetched_metadata:
            logger.info("Successfully fetched metadata from TMDb.")
            # Update the default metadata with the fetched data
            video_metadata.update(fetched_metadata)
        else:
            logger.warning(f"Could not fetch metadata for '{video_title}'. Proceeding with title only.")
    else:
        logger.info("No --title provided. Skipping automatic metadata fetching.")
    
    metadata_path = os.path.join(video_specific_dir, 'video_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(video_metadata, f, indent=2)
    logger.info(f"    -> Video metadata saved to {metadata_path}")

    # 1. Detect shots first to create the data "skeleton"
    scenes = []
    if not os.path.exists(paths["shots"]):
        scenes = detect_shot_boundaries(video_path, paths["shots"])
    else:
        logger.info(f"    -> Skipping shot detection, loading from {paths['shots']}.")
        with open(paths["shots"], 'r') as f:
            scenes = json.load(f)

    # 2. Extract audio
    if not os.path.exists(paths["audio"]):
        extract_audio(video_path, paths["audio"])

    # 3. Create raw transcript
    if not os.path.exists(paths["transcript_raw"]):
        transcribe_and_diarize(paths["audio"], paths["transcript_raw"], CONFIG)

    # 4. Align the transcript to the shots
    if not os.path.exists(paths["transcript_aligned"]):
        align_transcript_to_shots(paths["transcript_raw"], scenes, paths["transcript_aligned"])
    
    # 5. Run per-shot analysis for audio events
    if not os.path.exists(paths["audio_events"]):
        detect_audio_events_per_shot(paths["audio"], scenes, paths["audio_events"], CONFIG)

    # 6. Run per-shot analysis for visual captions
    if not os.path.exists(paths["visual_details"]):
        generate_visual_captions(video_path, scenes, paths["visual_details"], CONFIG)

    if not os.path.exists(paths["actions"]):
        detect_actions_per_shot(video_path, scenes, paths["actions"], CONFIG)
    
    if not os.path.exists(paths["final_analysis"]):
        create_final_analysis_file(paths)

    logger.info(f"--- Extraction Complete for '{video_filename}'! ---")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run the data extraction pipeline using settings from config.yaml.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--output_dir", default="data/processed", help="Base directory to save processed subdirectories.")
    
    parser.add_argument("--title", help="Optional: The title of the movie to search for metadata.")
    parser.add_argument("--year", type=int, help="Optional: The release year of the movie for a more accurate search.")

    args = parser.parse_args()
    
    run_extraction(args.video, args.output_dir, args.title, args.year)
