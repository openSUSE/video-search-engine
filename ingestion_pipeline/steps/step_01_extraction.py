import subprocess
import whisperx
import torch
import logging
import os
import json
import csv
import cv2
from PIL import Image
import logging
from typing import Dict, Any, List
from core.config import CONFIG

logger = logging.getLogger(__name__)

def _get_paths(processed_dir: str, config: Dict[str, Any]) -> dict:
    """Generates a dictionary of all required output paths using filenames from config."""
    f_names = config['filenames']
    return {
        "audio": os.path.join(processed_dir, f_names['audio']),
        "transcript": os.path.join(processed_dir, f_names['transcript']),
        "shots": os.path.join(processed_dir, f_names['shots']),
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

def transcribe_and_diarize(audio_path: str, transcript_path: str, config: Dict[str, Any]):
    """Transcribes audio and performs speaker diarization using settings from config."""
    logger.info("    -> Transcribing and identifying speakers...")
    device = config['general']['device']
    model_cfg = config['models']['transcription']
    params_cfg = config['parameters']['transcription']
    
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model(model_cfg['name'], device, compute_type=model_cfg['compute_type'])
    result_transcript = model.transcribe(audio, batch_size=params_cfg['batch_size'])

    # Diarization - uses the token from config
    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=config['general']['hf_token'], device=device)
    diarize_segments = diarize_model(audio)
    result_transcript = whisperx.assign_word_speakers(diarize_segments, result_transcript)

    with open(transcript_path, 'w') as f:
        json.dump(result_transcript['segments'], f, indent=2)
    logger.info(f"    -> Transcript saved to {transcript_path}")

def detect_shot_boundaries(video_path: str, shots_path: str) -> list:
    """Detects shot boundaries using TransNetV2."""
    logger.info("    -> Detecting shot boundaries with TransNetV2...")
    from transnetv2_pytorch import TransNetV2
    model_transnet = TransNetV2()
    _, _, all_frame_predictions = model_transnet.predict_video(video_path)
    scenes = model_transnet.predictions_to_scenes(all_frame_predictions.cpu().numpy()).tolist()
    
    with open(shots_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["start_frame", "end_frame"])
        writer.writerows(scenes)
    logger.info(f"    -> Shot boundaries saved to {shots_path}")
    return scenes

def detect_audio_events_per_shot(video_path: str, audio_path: str, scenes: list, output_path: str, config: Dict[str, Any]):
    """Detects audio events for each shot using settings from config."""
    logger.info("    -> Detecting audio events per shot...")
    device = config['general']['device']
    model_cfg = config['models']['audio_events']
    audio_params = config['parameters']['audio']
    event_params = config['parameters']['audio_events']

    try:
        from transformers import AutoProcessor, AutoModelForAudioClassification
        import librosa

        processor = AutoProcessor.from_pretrained(model_cfg['name'])
        model = AutoModelForAudioClassification.from_pretrained(model_cfg['name']).to(device)
        
        sr = audio_params['sample_rate']
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        all_shot_events = []
        for i, (start_frame, end_frame) in enumerate(scenes):
            start_time = start_frame / fps
            end_time = end_frame / fps
            
            audio_chunk = y[int(start_time * sr):int(end_time * sr)]
            shot_events_info = {"shot_number": i + 1, "start_frame": int(start_frame), "end_frame": int(end_frame), "events": []}

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
    except Exception as e:
        logger.error(f"Failed to detect audio events: {e}", exc_info=True)

def generate_visual_captions(video_path: str, scenes: list, output_path: str, config: Dict[str, Any]):
    """Generates captions for each shot using settings from config."""
    logger.info("    -> Generating visual captions for shots...")
    device = config['general']['device']
    model_cfg = config['models']['visual_captioning']
    params_cfg = config['parameters']['visual_captioning']

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration

        processor = BlipProcessor.from_pretrained(model_cfg['name'])
        model = BlipForConditionalGeneration.from_pretrained(model_cfg['name']).to(device)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        visual_details = []
        for i, (start, end) in enumerate(scenes):
            middle_frame_idx = (start + end) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, frame = cap.read()

            if ret:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = processor(pil_image, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=params_cfg['max_new_tokens'])
                caption = processor.decode(out[0], skip_special_tokens=True)
                visual_details.append({"shot_number": i + 1, "start_frame": int(start), "end_frame": int(end), "caption": caption})
        cap.release()

        with open(output_path, 'w') as f:
            json.dump(visual_details, f, indent=2)
        logger.info(f"    -> Visual details saved.")
    except Exception as e:
        logger.error(f"Failed to extract visual details: {e}", exc_info=True)

# --- MAIN ORCHESTRATOR FUNCTION ---

def run_extraction(video_path: str, base_output_dir: str):
    """Runs the full data extraction pipeline for a given video."""
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    video_specific_dir = os.path.join(base_output_dir, video_filename)
    
    logger.info(f"--- Starting Step 1: Data Extraction for '{video_filename}' ---")
    logger.info(f"Output will be saved in: {video_specific_dir}")
    
    os.makedirs(video_specific_dir, exist_ok=True)
    paths = _get_paths(video_specific_dir, CONFIG)

    # All steps now use the global CONFIG object
    if not os.path.exists(paths["audio"]):
        extract_audio(video_path, paths["audio"])
    
    if not os.path.exists(paths["transcript"]):
        transcribe_and_diarize(paths["audio"], paths["transcript"], CONFIG)

    scenes = []
    if not os.path.exists(paths["shots"]):
        scenes = detect_shot_boundaries(video_path, paths["shots"])
    else:
        logger.info(f"    -> Skipping shot detection, loading from file.")
        with open(paths["shots"], 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            scenes = [(int(row[0]), int(row[1])) for row in reader]

    if not scenes: # Ensure scenes are loaded if any downstream step needs them
        with open(paths["shots"], 'r') as f:
            reader = csv.reader(f); 
            next(reader); 
            scenes = [(int(r[0]), int(r[1])) for r in reader]

    if not os.path.exists(paths["audio_events"]):
        detect_audio_events_per_shot(video_path, paths["audio"], scenes, paths["audio_events"], CONFIG)

    if not os.path.exists(paths["visual_details"]):
        generate_visual_captions(video_path, scenes, paths["visual_details"], CONFIG)

    logger.info(f"--- Extraction Complete for '{video_filename}'! ---")


if __name__ == '__main__':
    # Your script originally had two `if __name__ == '__main__':` blocks. I've merged them.
    import argparse
    parser = argparse.ArgumentParser(description="Run the data extraction pipeline using settings from config.yaml.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--output_dir", default="data/processed", help="Base directory to save processed subdirectories.")
    args = parser.parse_args()
    
    run_extraction(args.video, args.output_dir)