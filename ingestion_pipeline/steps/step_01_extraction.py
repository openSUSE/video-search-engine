# ingestion_pipeline/steps/step_01_extraction.py

import subprocess
import whisperx
import torch
import logging
import os
import json
import logging
import cv2 # <-- Added for frame extraction
from PIL import Image # <-- Added for image processing

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
    audio_events_path = os.path.join(processed_dir, "audio_events.json")
    visual_details_path = os.path.join(processed_dir, "visual_details.json")

    # --- 1. EXTRACT AND NORMALIZE AUDIO WITH FFMPEG ---
    logger.info("1/5: Checking for existing audio file...")
    if not os.path.exists(audio_path):
        logger.info("    -> Extracting and normalizing audio...")
        command = [
            'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'libmp3lame', audio_path
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info(f"    -> Audio saved to {audio_path}")
        except subprocess.CalledProcessError as e:
            logger.error("FFmpeg failed to process the audio.")
            logger.error(f"FFmpeg stderr:\n{e.stderr}")
            raise
    else:
        logger.info(f"    -> Skipping audio extraction. File already exists: {audio_path}")


    # --- 2. TRANSCRIBE AND DIARIZE WITH WHISPERX ---
    logger.info("2/5: Checking for existing transcript...")
    if not os.path.exists(transcript_path):
        logger.info("    -> Transcribing and identifying speakers...")
        # (Your WhisperX code remains commented out as in the original)
        audio = whisperx.load_audio(audio_path)
        model = whisperx.load_model("base", DEVICE, compute_type="int8")
        result_transcript = model.transcribe(audio, batch_size=32)
        
        # Diarization (speaker identification)
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token="REMOVED_HF_TOKEN", device=DEVICE)
        diarize_segments = diarize_model(audio)
        result_transcript = whisperx.assign_word_speakers(diarize_segments, result_transcript)
        
        with open(transcript_path, 'w') as f:
            json.dump(result_transcript['segments'], f, indent=2)
        logger.info(f"    -> Transcript saved to {transcript_path}")
    else:
        logger.info(f"    -> Skipping transcription. File already exists: {transcript_path}")


    # --- 3. DETECT SHOT BOUNDARIES WITH TRANSNETV2 ---
    logger.info("3/5: Detecting shot boundaries...")
    scenes = [] # Initialize scenes list
    if not os.path.exists(shots_path):
        logger.info("    -> Running TransNetV2 model...")
        from transnetv2_pytorch import TransNetV2
        model_transnet = TransNetV2()
        video_frames, single_frame_predictions, all_frame_predictions = model_transnet.predict_video(video_path)
        
        predictions_np = all_frame_predictions.cpu().numpy()
        scenes = model_transnet.predictions_to_scenes(predictions_np).tolist() # .tolist() for consistency
        with open(shots_path, 'w') as f:
            f.write("start_frame,end_frame\n")
            for start, end in scenes:
                f.write(f"{start},{end}\n")
        logger.info(f"    -> Shot boundaries saved to {shots_path}")
    else:
        logger.info(f"    -> Skipping shot detection. Loading from existing file: {shots_path}")
        import csv
        with open(shots_path, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                scenes.append((int(row[0]), int(row[1])))
        logger.info(f"    -> Loaded {len(scenes)} shots from file.")


    # --- 4. DETECT AUDIO EVENTS PER SHOT ---
    logger.info("4/5: Checking for existing audio events file...")
    if not os.path.exists(audio_events_path):
        logger.info("    -> Detecting audio events per shot...")
        try:
            from transformers import AutoProcessor, AutoModelForAudioClassification
            import librosa

            processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", use_fast=True)
            model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(DEVICE)
            
            sr = 16000
            y, _ = librosa.load(audio_path, sr=sr, mono=True)
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            all_shot_events = []
            for i, (start_frame, end_frame) in enumerate(scenes):
                start_time = start_frame / fps
                end_time = end_frame / fps
                
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                audio_chunk = y[start_sample:end_sample]

                shot_events_info = {
                    "shot_number": i + 1,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "start_time": round(start_time, 3),
                    "end_time": round(end_time, 3),
                    "events": []
                }

                if audio_chunk.shape[0] > 0:
                    inputs = processor(audio_chunk, sampling_rate=sr, return_tensors="pt").to(DEVICE)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    
                    scores = torch.sigmoid(logits[0]).cpu().numpy()
                    top_k = 3
                    top_indices = scores.argsort()[-top_k:][::-1]
                    
                    detected_events = []
                    for j in top_indices:
                        score = round(float(scores[j]), 3)
                        if score > 0.1:
                            label = model.config.id2label[j]
                            detected_events.append({"event": label, "score": score})
                    shot_events_info["events"] = detected_events
                
                all_shot_events.append(shot_events_info)

            with open(audio_events_path, 'w') as f:
                json.dump(all_shot_events, f, indent=2)
            logger.info(f"    -> Timestamped audio events saved to {audio_events_path}")

        except Exception as e:
            logger.error(f"Failed to detect audio events: {e}")
            pass
    else:
        logger.info(f"    -> Skipping audio event detection. File already exists: {audio_events_path}")


    # --- 5. EXTRACT VISUAL DETAILS (IMAGE CAPTIONING) ---
    logger.info("5/5: Checking for existing visual details file...")
    if not os.path.exists(visual_details_path):
        logger.info("    -> Extracting visual details...")
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration

            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError("Cannot open video file")

            visual_details = []
            for i, (start, end) in enumerate(scenes):
                middle_frame_idx = (start + end) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
                ret, frame = cap.read()

                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)

                    inputs = processor(pil_image, return_tensors="pt").to(DEVICE)
                    out = model.generate(**inputs, max_new_tokens=50)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    
                    visual_details.append({ 
                        "shot_number": i + 1,
                        "start_frame": int(start),
                        "end_frame": int(end),
                        "caption": caption
                    })

            cap.release()

            with open(visual_details_path, 'w') as f:
                json.dump(visual_details, f, indent=2)
            logger.info(f"    -> Visual details saved to {visual_details_path}")

        except Exception as e:
            logger.error(f"Failed to extract visual details: {e}")
            pass
    else:
        logger.info(f"    -> Skipping visual details extraction. File already exists: {visual_details_path}")

    logger.info("--- Extraction Complete! ---")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run the initial data extraction pipeline.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    args = parser.parse_args()
    run_extraction(args.video)