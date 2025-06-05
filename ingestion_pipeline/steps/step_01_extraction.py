import subprocess
import whisperx
import torch
import logging
import os
import json
import logging
import cv2
from PIL import Image

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
    logger.info("1/5: Extracting and normalizing audio...")
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

    # --- 2. TRANSCRIBE AND DIARIZE WITH WHISPERX ---
    logger.info("2/5: Transcribing and identifying speakers...")
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

    # --- NEW: 3. DETECT AUDIO EVENTS ---
    logger.info("3/5: Detecting audio events...")
    try:
        from transformers import AutoProcessor, AutoModelForAudioClassification
        import librosa

        # Lazily load the audio event detection model
        processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(DEVICE)
        
        # Load audio with librosa to ensure correct sampling rate
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Process the audio and run inference
        inputs = processor(y, sampling_rate=sr, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        scores = torch.sigmoid(logits[0]).cpu().numpy()
        
        # Get the top 5 detected events
        top_k = 5
        top_indices = scores.argsort()[-top_k:][::-1]
        
        detected_events = []
        for i in top_indices:
            score = round(float(scores[i]), 3)
            if score > 0.1: # Only include events with a reasonable confidence score
                label = model.config.id2label[i]
                detected_events.append({"event": label, "score": score})

        with open(audio_events_path, 'w') as f:
            json.dump(detected_events, f, indent=2)
        logger.info(f"    -> Audio events saved to {audio_events_path}")

    except Exception as e:
        logger.error(f"Failed to detect audio events: {e}")
        # This step is not critical, so we log the error but don't stop the script
        pass


    # --- 4. DETECT SHOT BOUNDARIES WITH TRANSNETV2 ---
    logger.info("4/5: Detecting shot boundaries...")
    from transnetv2_pytorch import TransNetV2
    model_transnet = TransNetV2()
    # Note: predict_video loads all frames, which can be memory intensive.
    video_frames, single_frame_predictions, all_frame_predictions = model_transnet.predict_video(video_path)
    
    predictions_np = all_frame_predictions.cpu().numpy()
    scenes = model_transnet.predictions_to_scenes(predictions_np)
    with open(shots_path, 'w') as f:
        f.write("start_frame,end_frame\n")
        for start, end in scenes:
            f.write(f"{start},{end}\n")
    logger.info(f"    -> Shot boundaries saved to {shots_path}")

    # --- NEW: 5. EXTRACT VISUAL DETAILS (IMAGE CAPTIONING) ---
    logger.info("5/5: Extracting visual details...")
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration

        # Lazily load the image captioning model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Cannot open video file")

        visual_details = []
        for i, (start, end) in enumerate(scenes):
            # Get the middle frame of the shot
            middle_frame_idx = (start + end) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, frame = cap.read()

            if ret:
                # Convert frame from BGR (OpenCV) to RGB (PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Generate caption
                inputs = processor(pil_image, return_tensors="pt").to(DEVICE)
                out = model.generate(**inputs, max_new_tokens=50)
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                visual_details.append({
                    "shot_number": i + 1,
                    "start_frame": start,
                    "end_frame": end,
                    "caption": caption
                })
        
        cap.release()

        with open(visual_details_path, 'w') as f:
            json.dump(visual_details, f, indent=2)
        logger.info(f"    -> Visual details saved to {visual_details_path}")

    except Exception as e:
        logger.error(f"Failed to extract visual details: {e}")
        # This step is not critical, so we log the error but don't stop the script
        pass


    logger.info("--- Extraction Complete! ---")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run the initial data extraction pipeline.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    args = parser.parse_args()
    run_extraction(args.video)