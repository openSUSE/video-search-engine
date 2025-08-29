# core/config.py
import os, logging, torch, yaml
from typing import Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _yaml(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_config() -> Dict[str, Any]:
    load_dotenv()  # helpful locally

    cfg = _yaml(os.getenv("CONFIG_PATH", "config.yaml"))

    # ensure required sections exist
    for k in [
        "general",
        "ui",
        "api_server",
        "database",
        "filenames",
        "models",
        "parameters",
        "llm_enrichment",
    ]:
        cfg.setdefault(k, {})

    # ------- device selection -------
    ml_dev = os.getenv("ML_DEVICE", cfg["general"].get("device", "auto")).lower()
    device = "cpu"
    if ml_dev == "cuda" or (ml_dev == "auto" and torch.cuda.is_available()):
        device = "cuda"
    cfg["general"]["device"] = device
    cfg["general"]["hf_token"] = os.getenv("HF_TOKEN", cfg["general"].get("hf_token"))
    cfg["general"]["default_output_dir"] = os.getenv(
        "OUTPUT_DIR", cfg["general"].get("default_output_dir", "data/processed")
    )

    # ------- networking -------
    cfg["ui"]["host"] = os.getenv("UI_HOST", cfg["ui"].get("host", "0.0.0.0"))
    cfg["ui"]["port"] = int(os.getenv("UI_PORT", cfg["ui"].get("port", 5050)))
    cfg["api_server"]["host"] = os.getenv(
        "API_HOST", cfg["api_server"].get("host", "0.0.0.0")
    )
    cfg["api_server"]["port"] = int(
        os.getenv("API_PORT", cfg["api_server"].get("port", 1234))
    )

    # ------- database -------
    cfg["database"]["host"] = os.getenv(
        "CHROMA_HOST", cfg["database"].get("host", "localhost")
    )
    cfg["database"]["port"] = int(
        os.getenv("CHROMA_PORT", cfg["database"].get("port", 8000))
    )
    cfg["database"]["collection_name"] = os.getenv(
        "CHROMA_COLLECTION",
        cfg["database"].get("collection_name", "video_search_engine"),
    )

    # ------- LLM provider overrides -------
    prov = os.getenv("LLM_PROVIDER", cfg["llm_enrichment"].get("provider", "gemini"))
    cfg["llm_enrichment"]["provider"] = prov
    cfg["llm_enrichment"].setdefault("ollama", {})
    cfg["llm_enrichment"]["ollama"]["host"] = os.getenv(
        "OLLAMA_HOST", cfg["llm_enrichment"]["ollama"].get("host", "http://localhost")
    )
    cfg["llm_enrichment"]["ollama"]["port"] = int(
        os.getenv("OLLAMA_PORT", cfg["llm_enrichment"]["ollama"].get("port", 11434))
    )
    cfg["llm_enrichment"]["ollama"]["model"] = os.getenv(
        "OLLAMA_MODEL", cfg["llm_enrichment"]["ollama"].get("model", "gemma:2b")
    )
    cfg["llm_enrichment"].setdefault("gemini", {})
    cfg["llm_enrichment"]["gemini"]["model"] = os.getenv(
        "GEMINI_MODEL", cfg["llm_enrichment"]["gemini"].get("model", "gemini-1.5-flash")
    )

    # optional: model cache dirs (faster restarts in k8s)
    # Optional model cache dirs:
    # Prefer explicit envs. Otherwise pick a writable default.
    def _ensure_dir(p: str) -> str:
        try:
            os.makedirs(p, exist_ok=True)
            return p
        except Exception:
            return None

    # If user set any of these, respect them.
    hf_home = os.getenv("HF_HOME")
    tf_cache = os.getenv("TRANSFORMERS_CACHE")
    st_home = os.getenv("SENTENCE_TRANSFORMERS_HOME")
    torch_home = os.getenv("TORCH_HOME")

    # Choose a sane, writable default root
    default_root = os.getenv("MODEL_CACHE_DIR") or os.path.join(os.getcwd(), ".models")
    if not _ensure_dir(default_root):
        # fallback to user's home cache
        default_root = os.path.expanduser("~/.cache")
        _ensure_dir(default_root)

    defaults = {
        "HF_HOME": hf_home or os.path.join(default_root, "hf"),
        "TRANSFORMERS_CACHE": tf_cache or os.path.join(default_root, "hf"),
        "SENTENCE_TRANSFORMERS_HOME": st_home or os.path.join(default_root, "hf"),
        "TORCH_HOME": torch_home or os.path.join(default_root, "torch"),
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)

    logger.info(f"Using device: {cfg['general']['device']}")
    return cfg


CONFIG = load_config()
