# ingestion_pipeline/steps/step_03_enrichment.py

import json
import logging
import os
import shutil
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

LLMClient = Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]

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


def _call_ollama_api(prompt: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ollama_config = config['llm_enrichment']['ollama']
    api_url = f"{ollama_config['host']}:{ollama_config['port']}/api/generate"
    payload = {
        "model": ollama_config['model'], "prompt": prompt,
        "stream": False, "format": "json"
    }
    try:
        import requests

        response = requests.post(api_url, json=payload, timeout=ollama_config.get('timeout_sec', 120))
        response.raise_for_status()
        return json.loads(response.json()['response'])
    except Exception as e:
        logger.error(f"Ollama API request failed: {e}")
        return None

def _call_gemini_api(prompt: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    gemini_config = config['llm_enrichment']['gemini']
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        logger.error("Gemini API request failed: GEMINI_API_KEY environment variable not set.")
        return None

    try:
        import google.generativeai as genai
        from google.api_core import exceptions as google_exceptions
    except ImportError as e:
        logger.error(f"Gemini API request failed: Google SDK is not installed: {e}")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(gemini_config['model'])
        generation_config = genai.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content(prompt, generation_config=generation_config)
        return json.loads(response.text)
    except (ValueError, google_exceptions.GoogleAPICallError, google_exceptions.RetryError) as e:
        logger.error(f"Gemini API request failed: {e}")
        return None


def _normalize_provider_name(provider: Any) -> str:
    return str(provider or "").strip().lower()


def _resolve_llm_client(
    provider: Any,
    llm_clients: Optional[Dict[str, LLMClient]],
) -> Optional[LLMClient]:
    provider_name = _normalize_provider_name(provider)

    if llm_clients:
        for client_name, client in llm_clients.items():
            if _normalize_provider_name(client_name) == provider_name:
                return client

    if provider_name == 'gemini':
        return _call_gemini_api
    if provider_name == 'ollama':
        return _call_ollama_api
    return None


def _video_synopsis(video_metadata: Dict[str, Any]) -> str:
    return (
        _clean_metadata_string(video_metadata.get('synopsis'))
        or _clean_metadata_string(video_metadata.get('logline'))
        or 'N/A'
    )


def _clean_metadata_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None

    value = value.strip()
    return value or None


def _load_video_metadata(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        logger.warning(f"{path} not found. Proceeding without video context.")
        return {}

    try:
        with open(path, 'r') as f:
            video_metadata = json.load(f)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse {path}. Proceeding without it.")
        return {}

    if not isinstance(video_metadata, dict):
        logger.warning(f"{path} must contain a JSON object. Proceeding without it.")
        return {}

    logger.info(f"Loaded video metadata from {path}")
    return video_metadata


def _validate_optional_string_field(
    segment: Dict[str, Any],
    index: int,
    field_name: str,
) -> None:
    if field_name in segment and not isinstance(segment[field_name], str):
        raise ValueError(
            f"segment at index {index} field {field_name} must be a string"
        )


def _validate_optional_string_list_field(
    segment: Dict[str, Any],
    index: int,
    field_name: str,
) -> None:
    if field_name not in segment:
        return

    value = segment[field_name]
    if not isinstance(value, list):
        raise ValueError(
            f"segment at index {index} field {field_name} must be a JSON array"
        )

    for item_index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(
                f"segment at index {index} field {field_name} item at index "
                f"{item_index} must be a string"
            )


def _validate_segments(segments: Any) -> list[Dict[str, Any]]:
    if not isinstance(segments, list):
        raise ValueError("segments file must contain a JSON array")

    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segment at index {index} must be a JSON object")

        segment_id = segment.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id.strip():
            raise ValueError(f"segment at index {index} must have a segment_id")

        _validate_optional_string_field(segment, index, "full_transcript")
        for field_name in (
            "speakers",
            "consolidated_visual_captions",
            "consolidated_actions",
            "consolidated_audio_events",
        ):
            _validate_optional_string_list_field(segment, index, field_name)

    return segments


def _load_segments_file(path: str) -> list[Dict[str, Any]]:
    with open(path, 'r') as f:
        return _validate_segments(json.load(f))


def _clean_llm_string(value: Any) -> Optional[str]:
    if value is None:
        return None

    cleaned = str(value).strip()
    return cleaned or None


def _normalize_llm_keywords(value: Any) -> list[str]:
    if isinstance(value, list):
        return [
            keyword
            for keyword in (_clean_llm_string(item) for item in value)
            if keyword
        ]

    keyword = _clean_llm_string(value)
    return [keyword] if keyword else []


def _safe_llm_updates(llm_data: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(llm_data, dict):
        return None

    updates: Dict[str, Any] = {}
    for field in ("title", "summary"):
        value = _clean_llm_string(llm_data.get(field))
        if value:
            updates[field] = value

    if "keywords" in llm_data:
        updates["keywords"] = _normalize_llm_keywords(llm_data["keywords"])

    return updates or None


def _has_complete_enrichment(segment: Dict[str, Any]) -> bool:
    title = _clean_llm_string(segment.get("title"))
    summary = _clean_llm_string(segment.get("summary"))
    keywords = segment.get("keywords")

    return bool(
        title
        and title != "Error"
        and summary
        and isinstance(keywords, list)
        and _normalize_llm_keywords(keywords)
    )


def run_enrichment(
    segments_path: str,
    config: Dict[str, Any],
    llm_clients: Optional[Dict[str, LLMClient]] = None,
) -> Optional[str]:
    """
    Enriches segments with LLM data, saving progress after each segment.
    On rerun, it skips already enriched segments.
    """
    processed_dir = os.path.dirname(segments_path)
    output_filename = config['filenames']['enriched_segments']
    output_path = os.path.join(processed_dir, output_filename)

    provider = _normalize_provider_name(
        config.get('llm_enrichment', {}).get('provider', 'ollama')
    )
    logger.info(f"--- Starting Step 3: LLM Enrichment using provider: '{provider}' ---")
    llm_client = _resolve_llm_client(provider, llm_clients)
    if llm_client is None:
        logger.error(f"Invalid LLM provider: {provider}. Halting.")
        return None

    # If the output file doesn't exist, create it by copying the input file.
    # This initializes our "state" file for enrichment.
    if not os.path.exists(output_path):
        logger.info(f"Output file not found. Creating initial version from {segments_path}.")
        try:
            _load_segments_file(segments_path)
            shutil.copy(segments_path, output_path)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to create initial output file: {e}")
            return None

    # Now, we read from and write to the same output_path.
    try:
        segments = _load_segments_file(output_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Could not read or parse the segments file at {output_path}: {e}")
        return None

    video_metadata_path = os.path.join(processed_dir, 'video_metadata.json')
    video_metadata = _load_video_metadata(video_metadata_path)

    video_title = _clean_metadata_string(video_metadata.get('title')) or 'N/A'
    video_synopsis = _video_synopsis(video_metadata)

    total_segments = len(segments)
    for i, segment in enumerate(segments):
        logger.info(f"  -> Checking segment {i+1}/{total_segments} ({segment['segment_id']})...")

        # Skip only complete prior enrichments; partial records are retried.
        if _has_complete_enrichment(segment):
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
        llm_data = llm_client(prompt, config)
        llm_updates = _safe_llm_updates(llm_data)

        # Update the segment in the list
        if llm_updates:
            segment.update(llm_updates)
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
