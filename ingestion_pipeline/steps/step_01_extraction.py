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
from core.config import CONFIG

logger = logging.getLogger(__name__)

# MODIFIED: Added a path for the raw (unaligned) transcript
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

# MODIFIED: This now saves a raw transcript, to be aligned later.
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

# MODIFIED: Now creates the rich shots.json file.
def detect_shot_boundaries(video_path: str, shots_path: str) -> List[Dict[str, Any]]:
    """Detects shot boundaries and saves them as a rich JSON object."""
    logger.info("    -> Detecting shot boundaries with TransNetV2...")
    from transnetv2_pytorch import TransNetV2
    
    # We need the video's FPS to calculate timestamps
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

# NEW: Function to align the transcript segments with the detected shots.
def align_transcript_to_shots(raw_transcript_path: str, scenes: List[Dict[str, Any]], aligned_transcript_path: str):
    """Aligns transcript segments to shots and saves the new transcript."""
    logger.info("    -> Aligning transcript to shots...")
    with open(raw_transcript_path, 'r') as f:
        transcript_segments = json.load(f)

    aligned_segments = []
    for segment in transcript_segments:
        if 'start' not in segment or 'end' not in segment:
            continue # Skip segments without timestamps

        segment_midpoint = (segment['start'] + segment['end']) / 2
        assigned_shot_id = None
        for shot in scenes:
            if shot['start_time_sec'] <= segment_midpoint < shot['end_time_sec']:
                assigned_shot_id = shot['shot_id']
                break
        
        # Create a new segment dictionary with the shot_id
        aligned_segment = {
            "start": segment.get('start'),
            "end": segment.get('end'),
            "text": segment.get('text', ''),
            "speaker": segment.get('speaker'),
            "shot_id": assigned_shot_id
        }
        aligned_segments.append(aligned_segment)

    with open(aligned_transcript_path, 'w') as f:
        json.dump(aligned_segments, f, indent=2)
    logger.info(f"    -> Aligned transcript saved to {aligned_transcript_path}")


# MODIFIED: Simplified to use shot_id and produce the new data structure.
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

# MODIFIED: Simplified to use shot_id and produce the new data structure.
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
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

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

# --- MAIN ORCHESTRATOR FUNCTION (MODIFIED LOGIC) ---
def run_extraction(video_path: str, base_output_dir: str):
    """Runs the full data extraction pipeline for a given video."""
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    video_specific_dir = os.path.join(base_output_dir, video_filename)
    
    logger.info(f"--- Starting Step 1: Data Extraction for '{video_filename}' ---")
    logger.info(f"Output will be saved in: {video_specific_dir}")
    
    os.makedirs(video_specific_dir, exist_ok=True)
    paths = _get_paths(video_specific_dir, CONFIG)

    # --- NEW PIPELINE ORDER ---

    # 1. Detect shots first to create the data "skeleton"
    scenes = []
    if not os.path.exists(paths["shots"]):
        scenes = detect_shot_boundaries(video_path, paths["shots"])
    else:
        logger.info(f"    -> Skipping shot detection, loading from {paths['shots']}.")
        with open(paths["shots"], 'r') as f:
            scenes = json.load(f)

    # 2. Extract audio (needed for transcription and audio events)
    if not os.path.exists(paths["audio"]):
        extract_audio(video_path, paths["audio"])

    # 3. Create raw transcript if it doesn't exist
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

    logger.info(f"--- Extraction Complete for '{video_filename}'! ---")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run the data extraction pipeline using settings from config.yaml.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--output_dir", default="data/processed", help="Base directory to save processed subdirectories.")
    args = parser.parse_args()
    
    run_extraction(args.video, args.output_dir)