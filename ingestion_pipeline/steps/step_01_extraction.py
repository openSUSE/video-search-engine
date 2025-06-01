# ingestion_pipeline/steps/step_01_extraction.py

import subprocess
import whisperx
import torch
import logging
import os
import json
import logging

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Check if a GPU is available, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
# On Apple Silicon (M-series chips), you can use "mps" for acceleration
# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")


def run_extraction(video_path: str):
    logger.info("--- Starting Step 1: Raw Data Extraction ---")

    # Define output paths and ensure the directory exists
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    audio_path = os.path.join(processed_dir, "normalized_audio.mp3")
    transcript_path = os.path.join(processed_dir, "transcript_generic.json")
    shots_path = os.path.join(processed_dir, "shots.csv")

    # # --- 2. EXTRACT AND NORMALIZE AUDIO WITH FFMPEG (WITH ERROR HANDLING) ---
    logger.info("1/3: Extracting and normalizing audio...")
    command = [
        'ffmpeg', '-y', '-i', video_path, '-vn', audio_path
    ]
    try:
        # We run the command and capture its output
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"   -> Audio saved to {audio_path}")
    except subprocess.CalledProcessError as e:
        # If the command fails, we log the detailed error from ffmpeg
        logger.error("FFmpeg failed to process the audio.")
        logger.error(f"FFmpeg stderr:\n{e.stderr}")
        # We stop the script because the next steps will fail without the audio
        raise  

    # # --- 3. TRANSCRIBE AND DIARIZE WITH WHISPERX ---
    logger.info("2/3: Transcribing and identifying speakers...")
    # Note: The first time you run this, it will download models which can take time.
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model("base", DEVICE, compute_type="int8")
    result = model.transcribe(audio, batch_size=32)
    
    # Diarization (speaker identification)
    # IMPORTANT: Replace YOUR_HF_TOKEN with your actual Hugging Face token
    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token="REMOVED_HF_TOKEN", device=DEVICE)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    # Save the result as a JSON file
    with open(transcript_path, 'w') as f:
        json.dump(result['segments'], f, indent=2)
    logger.info(f"   -> Transcript saved to {transcript_path}")

    # --- 4. DETECT SHOT BOUNDARIES WITH TRANSNETV2 ---
    logger.info("3/3: Detecting shot boundaries...")
    # Lazily import TransNetV2 to avoid loading it if ffmpeg fails
    from transnetv2_pytorch import TransNetV2
    model = TransNetV2()
    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)
    
    # Save scenes to a simple CSV file
    predictions_np = all_frame_predictions.cpu().numpy()
    scenes = model.predictions_to_scenes(predictions_np)
    with open(shots_path, 'w') as f:
        f.write("start_frame,end_frame\n")
        for start, end in scenes:
            f.write(f"{start},{end}\n")
    logger.info(f"   -> Shot boundaries saved to {shots_path}")

    logger.info("--- Extraction Complete! ---")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run the initial data extraction pipeline.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    args = parser.parse_args()
    
    run_extraction(args.video)

