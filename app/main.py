import argparse
import json
import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from core.config import CONFIG # Import the global config object

# Basic configuration for Flask logging
logging.basicConfig(level=logging.INFO)
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
    logger.info(f"Serving video: {video_filename} from {video_dir}")
    return send_from_directory(video_dir, video_filename)

@app.route('/api/data')
def get_data():
    """API endpoint to provide the raw transcript data to the frontend."""
    try:
        with open(TRANSCRIPT_PATH, 'r') as f:
            transcript = json.load(f)
        return jsonify({
            "transcript": transcript,
            "video_filename": os.path.basename(VIDEO_PATH)
        })
    except FileNotFoundError:
        logger.error(f"Transcript file not found at: {TRANSCRIPT_PATH}")
        return jsonify({"error": "Transcript file not found."}), 404
    except Exception as e:
        logger.error(f"Error reading transcript file: {e}", exc_info=True)
        return jsonify({"error": "Could not read transcript file."}), 500


@app.route('/api/save_map', methods=['POST'])
def save_speaker_map():
    """API endpoint to receive speaker mappings and save them to a file."""
    data = request.json
    speaker_map = data.get('speaker_map')
    
    if not speaker_map:
        return jsonify({"error": "No speaker map data received."}), 400

    video_filename_no_ext = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    
    # Use the filename from the central CONFIG file
    speaker_map_filename = CONFIG['filenames']['speaker_map']
    output_path = os.path.join(OUTPUT_DIR, video_filename_no_ext, speaker_map_filename)
    
    # Ask the server to shut down in the background — UI is done.
    try:
        import threading, requests
        def _shutdown():
            try:
                requests.post(f"http://{CONFIG['ui']['host']}:{CONFIG['ui']['port']}/api/shutdown", timeout=2)
            except Exception:
                pass
        threading.Thread(target=_shutdown, daemon=True).start()
    except Exception:
        pass
    return jsonify({"message": "Speaker map saved successfully. Shutting down UI..."})

def shutdown_server():
    """Function to shut down the Werkzeug server."""
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        logger.warning("Could not shut down server automatically. The pipeline will continue.")
        return
    func()

@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    """Shuts down the server, called from the frontend after a successful save."""
    logger.info("Server is shutting down...")
    shutdown_server()
    return "Server is shutting down."

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Speaker ID UI Server.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--transcript", required=True, help="Path to the raw transcript JSON file.")
    parser.add_argument("--output_dir", required=True, help="Base directory to save the speaker_map.json.")
    
    # Add port argument, with the default pulled from the CONFIG
    parser.add_argument(
        "--port", 
        type=int, 
        default=CONFIG['ui']['port'], 
        help="Port to run the server on."
    )
    args = parser.parse_args()

    VIDEO_PATH = os.path.abspath(args.video)
    TRANSCRIPT_PATH = os.path.abspath(args.transcript)
    OUTPUT_DIR = os.path.abspath(args.output_dir)

    # Get host and port from CONFIG
    host = CONFIG['ui']['host']
    port = args.port # Use the port from command-line args

    logger.info(f"Starting server for video: {VIDEO_PATH}")
    logger.info(f"Please open http://{host}:{port} in your browser.")

    # Run the app with host and port from config/args
    app.run(host=host, port=port, debug=False, use_reloader=False)
