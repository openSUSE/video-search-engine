# core/config.py

import os
import logging
import torch
from typing import Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """
    Loads configuration by prioritizing environment variables, making it
    suitable for containerized deployments. Falls back to sane defaults
    for local development.
    """
    # load_dotenv() is useful for local development, allowing you to use a .env file
    # In Kubernetes, environment variables will be set directly.
    if load_dotenv():
        logger.info("Loaded environment variables from .env file for local development.")

    config = {}

    # --- General settings ---
    device = "cpu"  # Default to CPU for cluster environments
    if os.getenv("ML_DEVICE", "auto").lower() == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
    elif os.getenv("ML_DEVICE", "cpu").lower() == 'cuda':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            logger.warning("CUDA device requested via ML_DEVICE env var, but not available. Falling back to CPU.")
    
    config['general'] = {
        "device": device,
        "hf_token": os.getenv("HF_TOKEN"), # Loaded from a K8s Secret
        "default_output_dir": os.getenv("OUTPUT_DIR", "data/processed") # Path inside the container
    }
    logger.info(f"Using device: {config['general']['device']}")

    # --- Server and Database connection details ---
    # These will be set by Kubernetes services and ConfigMaps
    config['ui'] = {
        "host": os.getenv("UI_HOST", "127.0.0.1"),
        "port": int(os.getenv("UI_PORT", 5050))
    }
    config['api_server'] = {
        "host": os.getenv("API_HOST", "127.0.0.1"),
        "port": int(os.getenv("API_PORT", 1234))
    }
    config['database'] = {
        "host": os.getenv("CHROMA_HOST", "localhost"),
        "port": int(os.getenv("CHROMA_PORT", 8000)),
        "collection_name": os.getenv("CHROMA_COLLECTION", "video_search_engine")
    }

    # --- LLM Enrichment Settings ---
    # The provider is configured via env var. Secrets (API keys) are also env vars.
    config['llm_enrichment'] = {
        "provider": os.getenv("LLM_PROVIDER", "gemini"),
        "ollama": {
            "host": os.getenv("OLLAMA_HOST", "http://localhost"),
            "port": int(os.getenv("OLLAMA_PORT", 11434)),
            "model": os.getenv("OLLAMA_MODEL", "gemma:2b")
        },
        "gemini": {
            "model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            # The Gemini API key itself is read directly from the GEMINI_API_KEY
            # environment variable by the google-generativeai library.
        }
    }
    
    # --- Filenames (can remain static, but could also be configured) ---
    config['filenames'] = {
        "audio": "normalized_audio.mp3",
        "raw_transcript": "transcript_raw.json",
        "speaker_map": "speaker_map.json",
        "transcript": "transcript_generic.json",
        "shots": "shots.json",
        "audio_events": "audio_events.json",
        "visual_details": "visual_details.json",
        "actions": "actions.json",
        "final_analysis": "final_analysis.json",
        "final_segments": "final_segments.json",
        "enriched_segments": "final_enriched_segments.json"
    }

    # --- Models (can be configured via env vars) ---
    config['models'] = {
        "embedding": {
            "name": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        }
        # ... add others as needed ...
    }

    return config

# Load config once on module import to be used globally
CONFIG = load_config()