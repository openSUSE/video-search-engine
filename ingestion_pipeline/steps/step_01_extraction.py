import subprocess
import whisperx
import torch
import logging
import os
import json
import csv
import cv2
from PIL import Image

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if a GPU is available, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# On Apple Silicon (M-series chips), you can use "mps" for acceleration
# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# --- HELPER FUNCTIONS ---

def _get_paths(processed_dir: str) -> dict:
    """Generates a dictionary of all required output paths."""
    return {
        "audio": os.path.join(processed_dir, "normalized_audio.mp3"),
        "transcript": os.path.join(processed_dir, "transcript_generic.json"),
        "shots": os.path.join(processed_dir, "shots.csv"),
        "audio_events": os.path.join(processed_dir, "audio_events.json"),
        "visual_details": os.path.join(processed_dir, "visual_details.json"),
    }

# --- SPECIALIZED WORKER FUNCTIONS ---

def extract_audio(video_path: str, audio_path: str):
    """
    Extracts audio from a video file and saves it as an MP3.
    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to save the output audio file.
    """
    logger.info("    -> Extracting and normalizing audio...")
    command = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'libmp3lame', audio_path]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"    -> Audio saved to {audio_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed to process the audio.\nFFmpeg stderr:\n{e.stderr}")
        raise

def transcribe_and_diarize(audio_path: str, transcript_path: str, device: str):
    """
    Transcribes audio and performs speaker diarization using WhisperX.
    Args:
        audio_path (str): Path to the input audio file.
        transcript_path (str): Path to save the output transcript JSON.
        device (str): The compute device to use ('cuda', 'cpu').
    """
    logger.info("    -> Transcribing and identifying speakers...")
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model("base", device, compute_type="int8")
    result_transcript = model.transcribe(audio, batch_size=32)

    # Diarization (speaker identification)
    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token="REMOVED_HF_TOKEN", device=device)
    diarize_segments = diarize_model(audio)
    result_transcript = whisperx.assign_word_speakers(diarize_segments, result_transcript)

    with open(transcript_path, 'w') as f:
        json.dump(result_transcript['segments'], f, indent=2)
    logger.info(f"    -> Transcript saved to {transcript_path}")

def detect_shot_boundaries(video_path: str, shots_path: str) -> list:
    """
    Detects shot boundaries using TransNetV2 and saves them to a CSV.
    Args:
        video_path (str): Path to the input video file.
        shots_path (str): Path to save the output shots CSV.
    Returns:
        list: A list of tuples, where each tuple is a (start_frame, end_frame) pair.
    """
    logger.info("    -> Running TransNetV2 model...")
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

def detect_audio_events_per_shot(video_path: str, audio_path: str, scenes: list, output_path: str, device: str):
    """
    Detects audio events for each shot using a pre-trained AST model.
    Args:
        video_path (str): Path to the video to get FPS.
        audio_path (str): Path to the full audio file.
        scenes (list): List of (start_frame, end_frame) tuples.
        output_path (str): Path to save the audio events JSON.
        device (str): The compute device to use ('cuda', 'cpu').
    """
    logger.info("    -> Detecting audio events per shot...")
    try:
        from transformers import AutoProcessor, AutoModelForAudioClassification
        import librosa

        processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device)
        
        sr = 16000
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        all_shot_events = []
        for i, (start_frame, end_frame) in enumerate(scenes):
            start_time = start_frame / fps
            end_time = end_frame / fps
            
            audio_chunk = y[int(start_time * sr):int(end_time * sr)]
            shot_events_info = { "shot_number": i + 1, "start_frame": int(start_frame), "end_frame": int(end_frame), "start_time": round(start_time, 3), "end_time": round(end_time, 3), "events": [] }

            if audio_chunk.shape[0] > 0:
                inputs = processor(audio_chunk, sampling_rate=sr, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                
                scores = torch.sigmoid(logits[0]).cpu().numpy()
                top_indices = scores.argsort()[-3:][::-1]
                
                detected_events = [{"event": model.config.id2label[j], "score": round(float(scores[j]), 3)} for j in top_indices if scores[j] > 0.1]
                shot_events_info["events"] = detected_events
            
            all_shot_events.append(shot_events_info)

        with open(output_path, 'w') as f:
            json.dump(all_shot_events, f, indent=2)
        logger.info(f"    -> Timestamped audio events saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to detect audio events: {e}", exc_info=True)
        pass

def generate_visual_captions(video_path: str, scenes: list, output_path: str, device: str):
    """
    Generates a caption for the middle frame of each shot using a BLIP model.
    Args:
        video_path (str): Path to the input video file.
        scenes (list): List of (start_frame, end_frame) tuples.
        output_path (str): Path to save the visual details JSON.
        device (str): The compute device to use ('cuda', 'cpu').
    """
    logger.info("    -> Extracting visual details...")
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

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
                out = model.generate(**inputs, max_new_tokens=50)
                caption = processor.decode(out[0], skip_special_tokens=True)
                visual_details.append({"shot_number": i + 1, "start_frame": int(start), "end_frame": int(end), "caption": caption})
        cap.release()

        with open(output_path, 'w') as f:
            json.dump(visual_details, f, indent=2)
        logger.info(f"    -> Visual details saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to extract visual details: {e}", exc_info=True)
        pass

# --- MAIN ORCHESTRATOR FUNCTION ---

def run_extraction(video_path: str, base_output_dir: str = "data/processed"):
    """
    Runs the full data extraction pipeline for a given video.
    Args:
        video_path (str): Path to the source video file.
        base_output_dir (str): The directory to save all processed files.
    """
    logger.info("--- Starting Step 1: Raw Data Extraction ---")
    os.makedirs(base_output_dir, exist_ok=True)
    paths = _get_paths(base_output_dir)

    # --- 1. EXTRACT AND NORMALIZE AUDIO ---
    logger.info("1/5: Checking for audio file...")
    if not os.path.exists(paths["audio"]):
        extract_audio(video_path, paths["audio"])
    else:
        logger.info(f"    -> Skipping audio extraction. File already exists: {paths['audio']}")

    # --- 2. TRANSCRIBE AND DIARIZE ---
    logger.info("2/5: Checking for transcript...")
    if not os.path.exists(paths["transcript"]):
        transcribe_and_diarize(paths["audio"], paths["transcript"], DEVICE)
    else:
        logger.info(f"    -> Skipping transcription. File already exists: {paths['transcript']}")

    # --- 3. DETECT SHOT BOUNDARIES ---
    logger.info("3/5: Checking for shot boundaries...")
    scenes = []
    if not os.path.exists(paths["shots"]):
        scenes = detect_shot_boundaries(video_path, paths["shots"])
    else:
        logger.info(f"    -> Skipping shot detection. Loading from existing file: {paths['shots']}")
        with open(paths["shots"], 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            scenes = [(int(row[0]), int(row[1])) for row in reader]
        logger.info(f"    -> Loaded {len(scenes)} shots from file.")

    # --- 4. DETECT AUDIO EVENTS PER SHOT ---
    logger.info("4/5: Checking for audio events file...")
    if not os.path.exists(paths["audio_events"]):
        detect_audio_events_per_shot(video_path, paths["audio"], scenes, paths["audio_events"], DEVICE)
    else:
        logger.info(f"    -> Skipping audio event detection. File already exists: {paths['audio_events']}")

    # --- 5. EXTRACT VISUAL DETAILS (IMAGE CAPTIONING) ---
    logger.info("5/5: Checking for visual details file...")
    if not os.path.exists(paths["visual_details"]):
        generate_visual_captions(video_path, scenes, paths["visual_details"], DEVICE)
    else:
        logger.info(f"    -> Skipping visual details extraction. File already exists: {paths['visual_details']}")

    logger.info("--- Extraction Complete! ---")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run the initial data extraction pipeline.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--output_dir", default="data/processed", help="Directory to save processed files.")
    args = parser.parse_args()
    
    run_extraction(args.video, args.output_dir)