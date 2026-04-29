import argparse
import json
import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory

from app.ui.speaker_support import (
    load_transcript_segments,
    load_transcript_speaker_ids,
    normalize_speaker_map,
)
from app.ui.url_settings import local_http_url
from core.atomic_io import atomic_write_json
from core.logger import setup_logging

# Honour LOG_LEVEL on this Flask UI subprocess too — setup_logging is
# idempotent so it's safe even if the parent already configured logging.
setup_logging()
logger = logging.getLogger(__name__)

# Initialize Flask app
# The static_folder path needs to be relative to the app's root.
# Assuming 'main.py' is in 'app/', and 'static' is in 'app/ui/'.
app = Flask(__name__, template_folder='ui', static_folder='ui/static')

# --- Global variables to hold paths from command line ---
# These are set in the `if __name__ == '__main__':` block
VIDEO_PATH = None
TRANSCRIPT_PATH = None
OUTPUT_DIR = None
CONFIG = None


def get_config():
    global CONFIG
    if CONFIG is None:
        from core.config import CONFIG as loaded_config

        CONFIG = loaded_config
    return CONFIG


def _ui_url() -> str:
    config = get_config()
    return local_http_url(config['ui']['host'], config['ui']['port'])


@app.route('/')
def index():
    """Serves the main UI page."""
    return render_template('index.html')

@app.route('/video_file')
def video_file():
    """Serves the video file to the HTML <video> tag."""
    # It's safer to serve files from an absolute path
    video_dir = os.path.dirname(VIDEO_PATH)
    video_filename = os.path.basename(VIDEO_PATH)
    logger.info("Serving video: %s from %s", video_filename, video_dir)
    return send_from_directory(video_dir, video_filename)

@app.route('/api/data')
def get_data():
    """API endpoint to provide the raw transcript data to the frontend."""
    try:
        transcript = load_transcript_segments(TRANSCRIPT_PATH)
        return jsonify({
            "transcript": transcript,
            "video_filename": os.path.basename(VIDEO_PATH)
        })
    except FileNotFoundError:
        logger.error("Transcript file not found at: %s", TRANSCRIPT_PATH)
        return jsonify({"error": "Transcript file not found."}), 404
    except (json.JSONDecodeError, ValueError):
        logger.exception("Invalid transcript file at %s", TRANSCRIPT_PATH)
        return jsonify({"error": "Could not read transcript file."}), 500
    except Exception:
        logger.exception("Error reading transcript file at %s", TRANSCRIPT_PATH)
        return jsonify({"error": "Could not read transcript file."}), 500


def _required_speaker_ids():
    if not TRANSCRIPT_PATH:
        return []

    return load_transcript_speaker_ids(TRANSCRIPT_PATH)


@app.route('/api/save_map', methods=['POST'])
def save_speaker_map():
    """API endpoint to receive speaker mappings and save them to a file."""
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "No speaker map data received."}), 400

    speaker_map = normalize_speaker_map(data.get('speaker_map'))
    if speaker_map is None:
        return jsonify({"error": "Speaker map must contain non-empty speaker names."}), 400

    try:
        required_speaker_ids = _required_speaker_ids()
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error("Could not validate speaker map against transcript: %s", e, exc_info=True)
        return jsonify({"error": "Could not validate speaker map against transcript."}), 500

    missing_speaker_ids = sorted(set(required_speaker_ids) - set(speaker_map))
    if missing_speaker_ids:
        return jsonify({
            "error": "Speaker map is missing names for: " + ", ".join(missing_speaker_ids)
        }), 400

    video_filename_no_ext = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    
    # Use the filename from the central CONFIG file
    config = get_config()
    speaker_map_filename = config['filenames']['speaker_map']
    output_path = os.path.join(OUTPUT_DIR, video_filename_no_ext, speaker_map_filename)

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        atomic_write_json(output_path, speaker_map)
        logger.info("Saved speaker map to %s", output_path)
    except Exception:
        logger.exception("Failed to save speaker map to %s", output_path)
        return jsonify({"error": "Could not save speaker map."}), 500

    # Ask the server to shut down in the background; the UI work is done.
    _request_shutdown_async()
    return jsonify({"message": "Speaker map saved successfully. Shutting down UI..."})


def _request_shutdown_async():
    try:
        import threading, requests
        shutdown_url = f"{_ui_url()}/api/shutdown"
        def _shutdown():
            try:
                requests.post(shutdown_url, timeout=2)
            except Exception:
                pass
        threading.Thread(target=_shutdown, daemon=True).start()
    except Exception:
        pass

def shutdown_server():
    """Stop the Werkzeug dev server.

    Werkzeug ≥2.2 removed the ``werkzeug.server.shutdown`` magic that older
    Flask docs use, so we schedule ``os._exit`` from a background thread
    after a short delay — long enough for Flask to flush the response, short
    enough that the pipeline's ``server_process.wait`` returns promptly.
    The speaker map has already been atomic-written before this point, so
    skipping interpreter cleanup is safe.
    """
    import threading
    import time

    def _kill():
        time.sleep(0.2)
        os._exit(0)

    threading.Thread(target=_kill, daemon=True).start()


@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    """Shuts down the server, called from the frontend after a successful save."""
    logger.info("Server is shutting down...")
    shutdown_server()
    return "Server is shutting down."

if __name__ == '__main__':
    config = get_config()
    parser = argparse.ArgumentParser(description="Run Speaker ID UI Server.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--transcript", required=True, help="Path to the raw transcript JSON file.")
    parser.add_argument("--output_dir", required=True, help="Base directory to save the speaker_map.json.")
    
    # Add port argument, with the default pulled from the CONFIG
    parser.add_argument(
        "--port", 
        type=int, 
        default=config['ui']['port'],
        help="Port to run the server on."
    )
    args = parser.parse_args()

    VIDEO_PATH = os.path.abspath(args.video)
    TRANSCRIPT_PATH = os.path.abspath(args.transcript)
    OUTPUT_DIR = os.path.abspath(args.output_dir)

    # Get host and port from CONFIG
    host = config['ui']['host']
    port = args.port # Use the port from command-line args

    logger.info("Starting server for video: %s", VIDEO_PATH)
    logger.info("Please open %s in your browser.", local_http_url(host, port))

    # Run the app with host and port from config/args
    app.run(host=host, port=port, debug=False, use_reloader=False)
