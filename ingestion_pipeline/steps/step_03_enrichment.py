# ingestion_pipeline/steps/step_03_enrichment.py

import logging
import json
import os
import requests
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

logger = logging.getLogger(__name__)

# This multi-modal prompt works well for both Ollama and Gemini
PROMPT_TEMPLATE = """
You are an expert video content analyst. Your task is to analyze a segment of a video using the multi-modal data provided below and generate a concise analysis in a structured JSON format.

**[Segment Context]**
- Speakers: {speakers}
- Key Visuals: {visuals}
- Background Audio Events: {audio_events}

**[Segment Transcript]**
"{transcript}"

**[Instructions]**
Based on ALL the context provided (transcript, visuals, and audio), generate the following:
1. "title": A short, descriptive title (5-10 words) that captures the main topic and action of the segment.
2. "summary": A concise, neutral summary (2-3 sentences) integrating what was said with what was shown and heard.
3. "keywords": A JSON array of 5-7 important keywords or short phrases representing the core concepts, objects, and actions.

**Output Format (Strictly JSON):**
"""

def _call_ollama_api(prompt: str, config: dict) -> dict:
    """Sends a prompt to the local Ollama API."""
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
    """Sends a prompt to the Google Gemini API."""
    gemini_config = config['llm_enrichment']['gemini']
    try:
        # Configure the SDK using the GOOGLE_API_KEY environment variable
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(gemini_config['model'])
        # Enforce JSON output
        generation_config = genai.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content(prompt, generation_config=generation_config)
        return json.loads(response.text)
    except (ValueError, google_exceptions.GoogleAPICallError, google_exceptions.RetryError) as e:
        logger.error(f"Gemini API request failed: {e}")
        return None

def run_enrichment(segments_path: str, config: dict) -> str:
    """
    Enriches each segment with an LLM-generated title, summary, and keywords
    using the provider specified in the config (Ollama or Gemini).
    """
    processed_dir = os.path.dirname(segments_path)
    output_filename = config['filenames']['enriched_segments']
    output_path = os.path.join(processed_dir, output_filename)

    if os.path.exists(output_path):
        logger.info(f"--- Skipping Step 3: LLM Enrichment. Output already exists at {output_path} ---")
        return output_path

    provider = config.get('llm_enrichment', {}).get('provider', 'ollama')
    logger.info(f"--- Starting Step 3: LLM Enrichment using provider: '{provider}' ---")

    try:
        with open(segments_path, 'r') as f:
            segments = json.load(f)
    except FileNotFoundError:
        logger.error(f"Segments file not found at {segments_path}. Cannot perform enrichment.")
        return None

    enriched_segments = []
    total_segments = len(segments)
    
    for i, segment in enumerate(segments):
        logger.info(f"  -> Processing segment {i+1}/{total_segments} ({segment['segment_id']})...")
        transcript = segment.get("full_transcript", "").strip()
        
        if not transcript:
            logger.warning(f"  -> Segment {segment['segment_id']} has no transcript. Skipping LLM call.")
            segment.update({'llm_title': "N/A (No transcript)", 'llm_summary': "", 'llm_keywords': []})
            enriched_segments.append(segment)
            continue
        
        context = {
            "transcript": transcript,
            "speakers": ", ".join(segment.get('speakers', [])) or "None",
            "visuals": "; ".join(segment.get('consolidated_visual_captions', [])) or "None",
            "audio_events": ", ".join(segment.get('consolidated_audio_events', [])) or "None"
        }
        prompt = PROMPT_TEMPLATE.format(**context)
        
        # Call the appropriate API based on the config
        llm_data = None
        if provider == 'gemini':
            llm_data = _call_gemini_api(prompt, config)
        elif provider == 'ollama':
            llm_data = _call_ollama_api(prompt, config)
        else:
            logger.error(f"Invalid LLM provider specified: {provider}. Halting enrichment.")
            break
            
        if llm_data:
            segment.update(llm_data)
            logger.info(f"  -> Successfully enriched segment {segment['segment_id']}.")
        else:
            logger.error(f"  -> Failed to enrich segment {segment['segment_id']}.")
            segment.update({'title': 'Error', 'summary': 'Failed to generate', 'keywords': []})

        enriched_segments.append(segment)

    processed_dir = os.path.dirname(segments_path)
    output_filename = config['filenames']['enriched_segments']
    output_path = os.path.join(processed_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(enriched_segments, f, indent=4)
        
    logger.info(f"--- Enrichment Step Complete! Saved to {output_path} ---")
    return output_path