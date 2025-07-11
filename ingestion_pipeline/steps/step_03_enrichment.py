# ingestion_pipeline/steps/step_03_enrichment.py

import logging
import json
import os
import requests
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

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


def _call_ollama_api(prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Sends a prompt to the Ollama API and returns the parsed JSON response."""
    llm_config = config['llm_enrichment']
    api_url = f"{llm_config['host']}:{llm_config['port']}/api/generate"
    payload = {
        "model": llm_config['model'],
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=llm_config.get('timeout_sec', 120))
        response.raise_for_status()
        response_data = response.json()
        json_content = json.loads(response_data['response'])
        return json_content
    except requests.exceptions.Timeout:
        logger.error("Ollama API request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {e}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON from Ollama response: {response.text}")
        return None

def run_enrichment(segments_path: str, config: Dict[str, Any]) -> str:
    """
    Enriches each segment with an LLM-generated title, summary, and keywords
    using multi-modal context (transcript, visuals, speakers, audio events).
    """
    if not config.get('llm_enrichment', {}).get('enabled', False):
        logger.warning("LLM Enrichment (Step 3) is disabled in config.yaml. Skipping.")
        return segments_path

    logger.info("--- Starting Step 3: Multi-Modal LLM-Powered Enrichment ---")

    try:
        with open(segments_path, 'r') as f:
            segments = json.load(f)
    except FileNotFoundError:
        logger.error(f"Segments file not found at {segments_path}. Cannot perform enrichment.")
        return None

    enriched_segments = []
    total_segments = len(segments)
    model_name = config['llm_enrichment']['model']
    logger.info(f"Enriching {total_segments} segments using Ollama model '{model_name}'...")

    for i, segment in enumerate(segments):
        logger.info(f"  -> Processing segment {i+1}/{total_segments} ({segment['segment_id']})...")
        
        # --- ASSEMBLE MULTI-MODAL CONTEXT ---
        transcript = segment.get("full_transcript", "").strip()
        
        # Handle cases where there is no transcript to avoid sending an empty prompt
        if not transcript:
            logger.warning(f"  -> Segment {segment['segment_id']} has no transcript. Skipping LLM call.")
            segment.update({'llm_title': "N/A (No transcript)", 'llm_summary': "", 'llm_keywords': []})
            enriched_segments.append(segment)
            continue
        
        speakers_list = segment.get('speakers', [])
        visuals_list = segment.get('consolidated_visual_captions', [])
        audio_list = segment.get('consolidated_audio_events', [])
        
        # Format the context for clean insertion into the prompt
        context = {
            "transcript": transcript,
            "speakers": ", ".join(speakers_list) if speakers_list else "None",
            "visuals": "; ".join(visuals_list) if visuals_list else "None",
            "audio_events": ", ".join(audio_list) if audio_list else "None"
        }

        prompt = PROMPT_TEMPLATE.format(**context)
        llm_data = _call_ollama_api(prompt, config)

        if llm_data:
            segment.update({
                'llm_title': llm_data.get('title', 'N/A'),
                'llm_summary': llm_data.get('summary', 'N/A'),
                'llm_keywords': llm_data.get('keywords', [])
            })
            logger.info(f"  -> Successfully enriched segment {segment['segment_id']}.")
        else:
            logger.error(f"  -> Failed to enrich segment {segment['segment_id']}. Using placeholder data.")
            segment.update({
                'llm_title': "Error: Failed to generate",
                'llm_summary': "Error: Failed to generate",
                'llm_keywords': []
            })
        enriched_segments.append(segment)

    processed_dir = os.path.dirname(segments_path)
    output_filename = config['filenames']['enriched_segments']
    output_path = os.path.join(processed_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(enriched_segments, f, indent=4)
        
    logger.info(f"--- Enrichment Step Complete! Saved {len(enriched_segments)} enriched segments to {output_path} ---")
    return output_path