import argparse
import logging
import os
import sys
import time
import subprocess

from core.logger import setup_logging
from ingestion_pipeline.steps.step_01_extraction import run_extraction
from ingestion_pipeline.steps.step_02_segmentation import run_segmentation
from ingestion_pipeline.steps.step_03_enrichment import run_enrichment
from core.config import CONFIG

setup_logging()
logger = logging.getLogger(__name__)

# ... (The wait_for_speaker_identification function remains unchanged) ...
def wait_for_speaker_identification(video_path: str, output_dir: str):
    """
    Launches the speaker identification UI and waits for the user to complete it.
    """
    logger.info("--- Starting Step 1.5: Manual Speaker Identification ---")
    
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    video_specific_dir = os.path.join(output_dir, video_filename)
    
    # Paths required by the UI server, now read from CONFIG
    raw_transcript_path = os.path.join(video_specific_dir, CONFIG['filenames']['raw_transcript'])
    speaker_map_path = os.path.join(video_specific_dir, CONFIG['filenames']['speaker_map'])

    logger.info(f"Raw transcript path: {raw_transcript_path}")

    if os.path.exists(speaker_map_path):
        logger.info(f"Speaker map already exists at {speaker_map_path}. Skipping UI.")
        return speaker_map_path

    if not os.path.exists(raw_transcript_path):
        logger.error(f"Raw transcript not found at {raw_transcript_path}. Cannot start UI.")
        return None

    # Command to run the Flask server
    command = [
        sys.executable, "-m", "app.main",
        "--video", video_path,
        "--transcript", raw_transcript_path,
        "--output_dir", output_dir,
        "--port", str(CONFIG['ui']['port'])
    ]

    try:
        server_process = subprocess.Popen(command)
        # Build URL from CONFIG
        ui_url = f"http://{CONFIG['ui']['host']}:{CONFIG['ui']['port']}"
        logger.warning("Speaker identification UI server started.")
        logger.warning(f"Please go to {ui_url} to identify speakers.")
        logger.warning("The pipeline will automatically continue once you save your changes in the UI.")
        
        while not os.path.exists(speaker_map_path):
            if server_process.poll() is not None:
                logger.error("UI server exited prematurely. Please check the logs.")
                return None
            time.sleep(2)

        logger.info("Speaker map file found! Resuming pipeline...")
        return speaker_map_path

    except Exception as e:
        logger.error(f"Failed to run speaker identification UI: {e}", exc_info=True)
        if 'server_process' in locals() and server_process.poll() is None:
            server_process.terminate()
        return None


def main():
    parser = argparse.ArgumentParser(description="Run the full video ingestion pipeline.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument(
        "--output_dir", 
        default=CONFIG['general']['default_output_dir'], 
        help="Base directory to save processed subdirectories."
    )
    args = parser.parse_args()

    logger.info(f"Starting ingestion pipeline for video: {args.video}")
    
    try:
        # Step 1: Data Extraction
        run_extraction(args.video, args.output_dir)

        # Step 1.5: Manual Speaker Identification
        speaker_map_path = wait_for_speaker_identification(args.video, args.output_dir)
        
        if not speaker_map_path:
            raise RuntimeError("Speaker identification step failed. Halting pipeline.")

        video_filename = os.path.splitext(os.path.basename(args.video))[0]
        video_specific_dir = os.path.join(args.output_dir, video_filename)
        
        final_analysis_path = os.path.join(video_specific_dir, CONFIG["filenames"]["final_analysis"])

        # Step 2: Intelligent Segmentation
        final_segments_path = run_segmentation(
            video_path=args.video,
            analysis_path=final_analysis_path,
            speaker_map_path=speaker_map_path
        )
        
        if not final_segments_path:
            logger.warning("Segmentation step did not produce an output file. Halting pipeline.")
            return

        # Step 3: LLM-Powered Enrichment (NEW STEP)
        enriched_segments_path = run_enrichment(final_segments_path, CONFIG)
        
        if enriched_segments_path and os.path.exists(enriched_segments_path):
            logger.info("🚀 Ingestion pipeline completed successfully!")
            logger.info(f"Final enriched segments are available at: {enriched_segments_path}")
        else:
            logger.warning("Enrichment step failed or was skipped.")
            logger.info(f"The last successful output is the un-enriched segments file: {final_segments_path}")

    except Exception as e:
        logger.critical(f"Pipeline failed with a critical error: {e}", exc_info=True)

if __name__ == '__main__':
    main()
