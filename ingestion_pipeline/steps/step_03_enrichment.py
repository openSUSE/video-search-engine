# ingestion_pipeline/steps/step_03_enrichment.py

import logging
import json
import os
import requests
import shutil
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

logger = logging.getLogger(__name__)

# Main prompt for when a transcript is available
PROMPT_WITH_TRANSCRIPT = """
You are an expert video content analyst. Your task is to analyze a segment of a video using the multi-modal data provided below and generate a concise analysis in a structured JSON format.

**[Overall Video Context]**
- Title: {video_title}
- Synopsis: {video_synopsis}

**[Segment Context]**
- Speakers: {speakers}
- Key Visuals: {visuals}
- Key Actions: {actions} 
- Background Audio Events: {audio_events}

**[Segment Transcript]**
"{transcript}"

**[Instructions]**
Based on ALL the context provided (overall video context, transcript, visuals, and audio), generate the following:
1. "title": A short, descriptive title (5-10 words) that captures the main topic and action of the segment.
2. "summary": A concise, neutral summary (2-3 sentences) integrating what was said with what was shown and heard.
3. "keywords": A JSON array of 5-7 important keywords or short phrases representing the core concepts, objects, and actions.

**Output Format (Strictly JSON):**
"""

PROMPT_NO_TRANSCRIPT = """
You are an expert video content analyst. Your task is to analyze a segment of a video using ONLY the visual and audio event data provided below. Generate a concise analysis in a structured JSON format. NOTE: There is no spoken transcript for this segment.

**[Overall Video Context]**
- Title: {video_title}
- Synopsis: {video_synopsis}

**[Segment Context]**
- Key Visuals: {visuals}
- Key Actions: {actions}
- Background Audio Events: {audio_events}

**[Instructions]**
Based on the visual and audio context, generate the following:
1. "title": A short, descriptive title (5-10 words) for the visual action.
2. "summary": A concise, neutral summary (2-3 sentences) describing what is shown and heard.
3. "keywords": A JSON array of 5-7 important keywords or short phrases representing the core objects, sounds, and actions.

**Output Format (Strictly JSON):**
"""


def _call_ollama_api(prompt: str, config: dict) -> dict:
    ollama_config = config['llm_enrichment']['ollama']
    api_url = f"{ollama_config['host']}:{ollama_config['port']}/api/generate"
    payload = {
        "model": ollama_config['model'], "prompt": prompt,
        "stream": False, "format": "json"
    }
    try:
        response = requests.post(api_url, json=payload, timeout=ollama_config.get('timeout_sec', 120))
        response.raise_for_status()
        return json.loads(response.json()['response'])
    except Exception as e:
        logger.error(f"Ollama API request failed: {e}")
        return None

def _call_gemini_api(prompt: str, config: dict) -> dict:
    gemini_config = config['llm_enrichment']['gemini']
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(gemini_config['model'])
        generation_config = genai.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content(prompt, generation_config=generation_config)
        return json.loads(response.text)
    except (ValueError, google_exceptions.GoogleAPICallError, google_exceptions.RetryError) as e:
        logger.error(f"Gemini API request failed: {e}")
        return None

def run_enrichment(segments_path: str, config: dict) -> str:
    """
    Enriches segments with LLM data, saving progress after each segment.
    On rerun, it skips already enriched segments.
    """
    processed_dir = os.path.dirname(segments_path)
    output_filename = config['filenames']['enriched_segments']
    output_path = os.path.join(processed_dir, output_filename)

    provider = config.get('llm_enrichment', {}).get('provider', 'ollama')
    logger.info(f"--- Starting Step 3: LLM Enrichment using provider: '{provider}' ---")

    # If the output file doesn't exist, create it by copying the input file.
    # This initializes our "state" file for enrichment.
    if not os.path.exists(output_path):
        logger.info(f"Output file not found. Creating initial version from {segments_path}.")
        try:
            shutil.copy(segments_path, output_path)
        except IOError as e:
            logger.error(f"Failed to create initial output file: {e}")
            return None

    # Now, we read from and write to the same output_path.
    try:
        with open(output_path, 'r') as f:
            segments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Could not read or parse the segments file at {output_path}: {e}")
        return None

    video_metadata_path = os.path.join(processed_dir, 'video_metadata.json')
    video_metadata = {}
    if os.path.exists(video_metadata_path):
        try:
            with open(video_metadata_path, 'r') as f:
                video_metadata = json.load(f)
            logger.info(f"Loaded video metadata from {video_metadata_path}")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse {video_metadata_path}. Proceeding without it.")
    else:
        logger.warning(f"{video_metadata_path} not found. Proceeding without video context.")
    
    video_title = video_metadata.get('title', 'N/A')
    video_synopsis = video_metadata.get('synopsis', 'N/A')

    total_segments = len(segments)
    for i, segment in enumerate(segments):
        logger.info(f"  -> Checking segment {i+1}/{total_segments} ({segment['segment_id']})...")

        # Skip if the segment is already enriched (and wasn't an error).
        if segment.get('title') and segment.get('title') != 'Error':
            logger.info(f"  -> Segment {segment['segment_id']} already enriched. Skipping.")
            continue

        logger.info(f"  -> Segment {segment['segment_id']} requires enrichment. Processing...")
        transcript = segment.get("full_transcript", "").strip()

        context = {
            "video_title": video_title,
            "video_synopsis": video_synopsis,
            "transcript": transcript,
            "speakers": ", ".join(segment.get('speakers', [])) or "None",
            "visuals": "; ".join(segment.get('consolidated_visual_captions', [])) or "None",
            "actions": ", ".join(segment.get('consolidated_actions', [])) or "None",
            "audio_events": ", ".join(segment.get('consolidated_audio_events', [])) or "None"
        }

        if transcript:
            prompt = PROMPT_WITH_TRANSCRIPT.format(**context)
        else:
            logger.warning(f"  -> Segment {segment['segment_id']} has no transcript. Using visual/audio prompt.")
            prompt = PROMPT_NO_TRANSCRIPT.format(**context)

        # Call the appropriate API
        llm_data = None
        if provider == 'gemini':
            llm_data = _call_gemini_api(prompt, config)
        elif provider == 'ollama':
            llm_data = _call_ollama_api(prompt, config)
        else:
            logger.error(f"Invalid LLM provider: {provider}. Halting.")
            break # Exit loop on invalid config

        # Update the segment in the list
        if llm_data:
            segment.update(llm_data)
            logger.info(f"  -> Successfully enriched segment {segment['segment_id']}.")
        else:
            logger.error(f"  -> Failed to enrich segment {segment['segment_id']}.")
            # Mark as an error so it can be retried on the next run
            segment.update({'title': 'Error', 'summary': 'Failed to generate', 'keywords': []})

        # CRITICAL CHANGE: Save the entire list back to the file after every single update.
        try:
            with open(output_path, 'w') as f:
                json.dump(segments, f, indent=4)
            logger.info(f"  -> Progress saved to {output_path}")
        except IOError as e:
            logger.error(f"FATAL: Could not write progress to {output_path}. Halting. Error: {e}")
            return None # Stop if we can't save

    logger.info(f"--- Enrichment Step Complete! Final data is in {output_path} ---")
    return output_path