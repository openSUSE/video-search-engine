import yaml
import os
import logging
import torch
from typing import Dict, Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the YAML configuration file, resolves special values, 
    and prioritizes environment variables for secrets.
    """
    if load_dotenv():
        logger.info("Loaded environment variables from .env file.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # --- Resolve Hugging Face Token ---
    # Prioritize environment variable for security
    hf_token_env = os.getenv("HF_TOKEN")
    if hf_token_env:
        logger.info("Loaded Hugging Face token from HF_TOKEN environment variable.")
        config['general']['hf_token'] = hf_token_env
    elif config.get('general', {}).get('hf_token', 'YOUR_TOKEN').startswith('YOUR_'):
        logger.warning("Hugging Face token is not set. Diarization will likely fail. "
                       "Set it in config.yaml or as an HF_TOKEN environment variable.")
        config['general']['hf_token'] = None

    # --- Resolve Compute Device ---
    if config['general']['device'] == 'auto':
        if torch.cuda.is_available():
            config['general']['device'] = 'cuda'
        # Add MPS check for Apple Silicon if needed
        # elif torch.backends.mps.is_available():
        #     config['general']['device'] = 'mps'
        else:
            config['general']['device'] = 'cpu'
    
    logger.info(f"Using device: {config['general']['device']}")
    
    return config

# Load config once on module import to be used globally
CONFIG = load_config()