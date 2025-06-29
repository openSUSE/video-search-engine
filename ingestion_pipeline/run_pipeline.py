import argparse
import logging
import json
import os
from core.logger import setup_logging
from ingestion_pipeline.steps.step_01_extraction import run_extraction
from ingestion_pipeline.steps.step_02_segmentation import run_segmentation
from core.config import CONFIG

# Assuming setup_logging() is defined in core.logger
# If not, you can add a basic configuration here.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

setup_logging()
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the full video ingestion pipeline.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--output_dir", default="data/processed", help="Base directory to save processed subdirectories.")
    args = parser.parse_args()

    logger.info(f"Starting ingestion pipeline for video: {args.video}")
    
    try:
        # Step 1: Extract raw data. This now returns a dictionary of paths.
        run_extraction(args.video, args.output_dir)

        # --- FAKE STEP 1.5: MANUAL SPEAKER ID ---
        # For now, we'll create a fake speaker map. Later, your UI will create this file.
        # This should be saved inside the specific video's output directory.
        video_filename = os.path.splitext(os.path.basename(args.video))[0]
        video_specific_dir = os.path.join(args.output_dir, video_filename)
        
        logger.warning("SIMULATING speaker identification step.")
        speaker_map = {"SPEAKER_00": "Sarah", "SPEAKER_01": "Mark", "SPEAKER_02": "John"}
        speaker_map_path = os.path.join(video_specific_dir, "speaker_map.json")
        with open(speaker_map_path, 'w') as f:
            json.dump(speaker_map, f)
        logger.info(f"   -> Fake speaker map saved to {speaker_map_path}")

        # Step 2: Run segmentation using the final analysis file from step 1.
        final_segments_path = run_segmentation(
            video_path=args.video,
            analysis_path=os.path.join(video_specific_dir, CONFIG["filenames"]["final_analysis"]),  # CORRECTED: Use the unified analysis file
            speaker_map_path=speaker_map_path
        )
        
        if final_segments_path:
            logger.info("Ingestion pipeline completed successfully!")
            logger.info(f"Final segments are available at: {final_segments_path}")
        else:
            logger.warning("Segmentation step did not produce an output file.")

    except Exception as e:
        logger.critical(f"Pipeline failed with a critical error: {e}", exc_info=True)

if __name__ == '__main__':
    main()
