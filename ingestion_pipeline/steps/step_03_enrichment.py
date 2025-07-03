import logging
import json
import os
import requests
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Analyze the following video transcript segment and provide a concise analysis in a structured JSON format.

**Transcript:**
"{transcript}"

**Instructions:**
Based on the transcript, generate the following:
1. "title": A short, descriptive title (5-10 words) that captures the main topic of the segment.
2. "summary": A concise, neutral summary (2-3 sentences) of the key information or events discussed.
3. "keywords": A JSON array of 5-7 important keywords or short phrases that represent the core topics.

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
        "format": "json"  # Use Ollama's built-in JSON mode
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
    Enriches each segment with an LLM-generated title, summary, and keywords.
    This function implements "Phase 3: LLM-Powered Enrichment".
    
    Args:
        segments_path: Path to the 'final_segments.json' file from Phase 2.
        config: The global configuration dictionary.
    
    Returns:
        The path to the final, enriched segments file, or None on failure.
    """
    if not config.get('llm_enrichment', {}).get('enabled', False):
        logger.warning("LLM Enrichment (Step 3) is disabled in config.yaml. Skipping.")
        return segments_path

    logger.info("--- Starting Step 3: LLM-Powered Enrichment ---")

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
        transcript = segment.get("full_transcript", "").strip()

        if not transcript:
            logger.warning(f"  -> Segment {segment['segment_id']} has no transcript. Skipping LLM call.")
            segment.update({'llm_title': "", 'llm_summary': "", 'llm_keywords': []})
            enriched_segments.append(segment)
            continue
        
        prompt = PROMPT_TEMPLATE.format(transcript=transcript)
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