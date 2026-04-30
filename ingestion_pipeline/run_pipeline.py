import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time

from core.logger import setup_logging
from app.ui.speaker_support import (
    load_transcript_speaker_ids,
    normalize_speaker_map,
    save_speaker_map_if_complete,
)
from app.ui.url_settings import local_http_url
from ingestion_pipeline.jobs import (
    normalize_optional_string,
    normalize_optional_year,
    normalize_required_string,
)

logger = logging.getLogger(__name__)


def _load_config():
    from core.config import CONFIG

    return CONFIG


def _load_pipeline_steps():
    from ingestion_pipeline.steps.step_01_extraction import run_extraction
    from ingestion_pipeline.steps.step_02_segmentation import run_segmentation
    from ingestion_pipeline.steps.step_03_enrichment import run_enrichment
    from ingestion_pipeline.steps.step_04_indexing import run_indexing

    return run_extraction, run_segmentation, run_enrichment, run_indexing


def _clean_env_value(value: str | None) -> str | None:
    if value is None:
        return None

    value = value.strip()
    return value or None


def _speaker_ui_mode() -> str:
    return (_clean_env_value(os.getenv("SPEAKER_UI_MODE")) or "external").lower()


def _speaker_map_timeout_seconds():
    raw_value = _clean_env_value(os.getenv("SPEAKER_MAP_TIMEOUT_SECONDS"))
    if raw_value is None:
        return None

    try:
        timeout_seconds = float(raw_value)
    except ValueError:
        logger.warning("Invalid SPEAKER_MAP_TIMEOUT_SECONDS=%r; waiting without a timeout.", raw_value)
        return None

    if not math.isfinite(timeout_seconds):
        logger.warning("Invalid SPEAKER_MAP_TIMEOUT_SECONDS=%r; waiting without a timeout.", raw_value)
        return None

    if timeout_seconds <= 0:
        return None

    return timeout_seconds


def _is_terminal_speaker_wait_reason(wait_reason: str | None) -> bool:
    return bool(
        wait_reason
        and wait_reason.startswith(
            (
                "malformed raw transcript:",
                "raw transcript is not valid JSON:",
            )
        )
    )


def _speaker_ui_url(config) -> str:
    ui_config = config["ui"]
    return local_http_url(ui_config["host"], ui_config["port"])


def _speaker_map_readiness(
    speaker_map_path: str,
    raw_transcript_path: str,
) -> tuple[bool, str | None]:
    try:
        required_speaker_ids = load_transcript_speaker_ids(raw_transcript_path)
    except FileNotFoundError as exc:
        return False, f"raw transcript is not available yet: {exc}"
    except json.JSONDecodeError as exc:
        return False, f"raw transcript is not valid JSON: {exc}"
    except ValueError as exc:
        return False, f"malformed raw transcript: {exc}"

    if not os.path.exists(speaker_map_path):
        if not required_speaker_ids:
            try:
                os.makedirs(os.path.dirname(speaker_map_path), exist_ok=True)
                if save_speaker_map_if_complete(
                    speaker_map_path,
                    {},
                    required_speaker_ids,
                ):
                    logger.info(
                        "Transcript has no speaker IDs; wrote empty speaker map to %s.",
                        speaker_map_path,
                    )
                    return True, None
            except OSError as exc:
                return False, f"could not create empty speaker map: {exc}"
        return False, f"{speaker_map_path} does not exist yet"

    try:
        with open(speaker_map_path, "r") as f:
            speaker_map_data = json.load(f)
    except FileNotFoundError as exc:
        return False, f"speaker map is not available yet: {exc}"
    except (json.JSONDecodeError, ValueError) as exc:
        return False, f"speaker map is not readable: {exc}"

    speaker_map = normalize_speaker_map(speaker_map_data)
    if speaker_map is None:
        return False, "speaker map must contain non-empty speaker names"

    missing_speaker_ids = sorted(set(required_speaker_ids) - set(speaker_map))
    if missing_speaker_ids:
        return False, "speaker map is missing names for: " + ", ".join(missing_speaker_ids)

    return True, None


def _wait_until_speaker_map_ready(
    speaker_map_path: str,
    raw_transcript_path: str,
    server_process=None,
) -> bool:
    timeout_seconds = _speaker_map_timeout_seconds()
    deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
    last_wait_reason = None

    while True:
        is_ready, wait_reason = _speaker_map_readiness(speaker_map_path, raw_transcript_path)
        if is_ready:
            return True

        if wait_reason != last_wait_reason:
            logger.info("Waiting for speaker map: %s", wait_reason)
            last_wait_reason = wait_reason

        if _is_terminal_speaker_wait_reason(wait_reason):
            logger.error("Cannot continue waiting for speaker map: %s", wait_reason)
            return False

        if server_process and server_process.poll() is not None:
            logger.error("UI server exited prematurely. Please check the logs.")
            return False

        if deadline and time.monotonic() >= deadline:
            logger.error(
                "Timed out waiting for speaker map at %s after %.0f seconds.",
                speaker_map_path,
                timeout_seconds,
            )
            return False

        sleep_seconds = 2
        if deadline:
            sleep_seconds = min(sleep_seconds, max(0.1, deadline - time.monotonic()))
        time.sleep(sleep_seconds)


def _terminate_server_process(server_process) -> None:
    if server_process.poll() is not None:
        return

    server_process.terminate()
    try:
        server_process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        server_process.kill()


def wait_for_speaker_identification(video_path: str, output_dir: str, config=None):
    """
    Launches the speaker identification UI and waits for the user to complete it.
    """
    config = config or _load_config()
    logger.info("--- Starting Step 1.5: Manual Speaker Identification ---")
    
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    video_specific_dir = os.path.join(output_dir, video_filename)
    
    # Paths required by the UI server, now read from CONFIG
    raw_transcript_path = os.path.join(video_specific_dir, config['filenames']['raw_transcript'])
    speaker_map_path = os.path.join(video_specific_dir, config['filenames']['speaker_map'])

    logger.info("Raw transcript path: %s", raw_transcript_path)

    if _speaker_ui_mode() == "external":
        logger.warning("External speaker UI mode: waiting for speaker_map.json to appear...")
        if not _wait_until_speaker_map_ready(speaker_map_path, raw_transcript_path):
            return None
        logger.info("Complete speaker map found; continuing.")
        return speaker_map_path

    if os.path.exists(speaker_map_path):
        is_ready, wait_reason = _speaker_map_readiness(speaker_map_path, raw_transcript_path)
        if is_ready:
            logger.info("Speaker map already exists at %s. Skipping UI.", speaker_map_path)
            return speaker_map_path
        logger.info("Existing speaker map is not complete; launching UI. Reason: %s", wait_reason)

    if not os.path.exists(raw_transcript_path):
        logger.error("Raw transcript not found at %s. Cannot start UI.", raw_transcript_path)
        return None

    # Command to run the Flask server
    command = [
        sys.executable, "-m", "app.main",
        "--video", video_path,
        "--transcript", raw_transcript_path,
        "--output_dir", output_dir,
        "--port", str(config['ui']['port'])
    ]

    try:
        server_process = subprocess.Popen(command)
        # Build URL from CONFIG
        ui_url = _speaker_ui_url(config)
        logger.warning("Speaker identification UI server started.")
        logger.warning("Please go to %s to identify speakers.", ui_url)
        logger.warning("The pipeline will automatically continue once you save your changes in the UI.")

        if not _wait_until_speaker_map_ready(
            speaker_map_path,
            raw_transcript_path,
            server_process=server_process,
        ):
            _terminate_server_process(server_process)
            return None

        logger.info("Speaker map file found! Requesting UI shutdown...")
        try:
            import requests

            requests.post(f"{ui_url}/api/shutdown", timeout=3)
        except Exception as e:
            logger.warning("Could not reach UI shutdown endpoint: %s", e)

        # Give the server a moment to exit gracefully
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("UI did not shut down gracefully; terminating process...")
            server_process.terminate()
            try:
                server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                logger.warning("UI still running; killing process...")
                server_process.kill()
        logger.info("Speaker identification UI stopped. Resuming pipeline...")
        return speaker_map_path

    except Exception:
        logger.exception("Failed to run speaker identification UI.")
        if 'server_process' in locals() and server_process.poll() is None:
            _terminate_server_process(server_process)
        return None


def run_pipeline(
    video_path: str,
    output_dir: str,
    title: str = None,
    year: int = None,
    config=None,
) -> bool:
    try:
        video_path = normalize_required_string(video_path, "video_path")
        output_dir = normalize_required_string(output_dir, "output_dir")
        title = normalize_optional_string(title, "title")
        year = normalize_optional_year(year)
    except ValueError as e:
        logger.critical("Invalid pipeline input: %s", e)
        return False

    if config is None:
        config = _load_config()
    run_extraction, run_segmentation, run_enrichment, run_indexing = _load_pipeline_steps()

    logger.info("Starting ingestion pipeline for video: %s", video_path)

    try:
        # Step 1: Data Extraction
        final_analysis_path = run_extraction(
            video_path,
            output_dir,
            title,
            year,
            config=config,
        )
        if not final_analysis_path or not os.path.exists(final_analysis_path):
            logger.warning(
                "Extraction step did not produce a final analysis file at %s. "
                "Halting pipeline.",
                final_analysis_path,
            )
            return False

        # Step 1.5: Manual Speaker Identification
        speaker_map_path = wait_for_speaker_identification(video_path, output_dir, config=config)
        
        if not speaker_map_path:
            raise RuntimeError("Speaker identification step failed. Halting pipeline.")

        video_filename = os.path.splitext(os.path.basename(video_path))[0]

        # Step 2: Intelligent Segmentation
        final_segments_path = run_segmentation(
            video_path=video_path,
            analysis_path=final_analysis_path,
            speaker_map_path=speaker_map_path,
            config=config,
        )
        
        if not final_segments_path:
            logger.warning("Segmentation step did not produce an output file. Halting pipeline.")
            return False

        # Step 3: LLM-Powered Enrichment
        enriched_segments_path = run_enrichment(final_segments_path, config)
        
        if not enriched_segments_path:
            logger.warning("Enrichment step failed or was skipped. Halting pipeline.")
            return False

        # Step 4: Indexing
        run_indexing(
            enriched_segments_path=enriched_segments_path,
            video_filename=video_filename, # Pass the filename for metadata
            config=config
        )
        
        logger.info("🚀 Ingestion pipeline completed successfully!")
        logger.info("Final enriched segments are available at: %s", enriched_segments_path)
        return True


    except Exception:
        logger.critical("Pipeline failed with a critical error.", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the full video ingestion pipeline.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument(
        "--output_dir",
        help="Base directory to save processed subdirectories. Defaults to configured output dir."
    )
    parser.add_argument("--title", help="Optional: The title of the movie to search for metadata.")
    parser.add_argument("--year", type=int, help="Optional: The release year of the movie for a more accurate search.")

    args = parser.parse_args()

    try:
        video_path = normalize_required_string(args.video, "video")
        output_dir = normalize_optional_string(args.output_dir, "output_dir")
        title = normalize_optional_string(args.title, "title")
        year = normalize_optional_year(args.year)
    except ValueError as exc:
        parser.error(str(exc))

    setup_logging()
    config = _load_config()
    succeeded = run_pipeline(
        video_path,
        output_dir or config['general']['default_output_dir'],
        title,
        year,
        config=config,
    )
    if not succeeded:
        sys.exit(1)

if __name__ == '__main__':
    main()
