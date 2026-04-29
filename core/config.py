# core/config.py
import logging
import os
from typing import Dict, Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

CONFIG_SECTIONS = [
    "general",
    "ui",
    "api_server",
    "database",
    "filenames",
    "models",
    "parameters",
    "llm_enrichment",
]


def _yaml(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r") as f:
            loaded_config = yaml.safe_load(f)
        if loaded_config is None:
            return {}
        if not isinstance(loaded_config, dict):
            raise ValueError(f"config file {path} must contain a YAML mapping")
        return loaded_config
    return {}


def _ensure_config_sections(cfg: Dict[str, Any]) -> None:
    for section_name in CONFIG_SECTIONS:
        section = cfg.setdefault(section_name, {})
        if section is None:
            cfg[section_name] = {}
            continue
        if not isinstance(section, dict):
            raise ValueError(f"config section '{section_name}' must be a mapping")


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def _string_setting(env_name: str, config_value: Any, default: str) -> str:
    return _clean_string(os.getenv(env_name)) or _clean_string(config_value) or default


def _int_setting(env_name: str, config_value: Any, default: int) -> int:
    return int(_string_setting(env_name, config_value, str(default)))


def _select_device(requested_device: str) -> str:
    requested = (requested_device or "auto").strip().lower()
    if requested in {"cpu", "cuda", "mps"}:
        return requested

    if requested != "auto":
        logger.warning("Unknown ML_DEVICE '%s'; falling back to cpu.", requested_device)
        return "cpu"

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return "mps"

    return "cpu"


def load_config() -> Dict[str, Any]:
    load_dotenv()  # helpful locally

    cfg = _yaml(_clean_string(os.getenv("CONFIG_PATH")) or "config.yaml")

    _ensure_config_sections(cfg)

    # ------- device selection -------
    ml_dev = _string_setting("ML_DEVICE", cfg["general"].get("device"), "auto")
    cfg["general"]["device"] = _select_device(ml_dev)

    hf_token = _clean_string(os.getenv("HF_TOKEN"))
    if cfg["general"].get("hf_token") and not hf_token:
        logger.warning("Ignoring general.hf_token from config file; set HF_TOKEN in the environment.")
    cfg["general"]["hf_token"] = hf_token

    cfg["general"]["default_output_dir"] = _string_setting(
        "OUTPUT_DIR",
        cfg["general"].get("default_output_dir"),
        "data/processed",
    )

    # ------- networking -------
    cfg["ui"]["host"] = _string_setting("UI_HOST", cfg["ui"].get("host"), "0.0.0.0")
    cfg["ui"]["port"] = _int_setting("UI_PORT", cfg["ui"].get("port"), 5050)
    cfg["api_server"]["host"] = _string_setting(
        "API_HOST",
        cfg["api_server"].get("host"),
        "0.0.0.0",
    )
    cfg["api_server"]["port"] = _int_setting(
        "API_PORT",
        cfg["api_server"].get("port"),
        1234,
    )

    # ------- database -------
    cfg["database"]["host"] = _string_setting(
        "CHROMA_HOST",
        cfg["database"].get("host"),
        "localhost",
    )
    cfg["database"]["port"] = _int_setting(
        "CHROMA_PORT",
        cfg["database"].get("port"),
        8000,
    )
    cfg["database"]["collection_name"] = _string_setting(
        "CHROMA_COLLECTION",
        cfg["database"].get("collection_name"),
        "video_search_engine",
    )

    # ------- LLM provider overrides -------
    prov = _string_setting(
        "LLM_PROVIDER",
        cfg["llm_enrichment"].get("provider"),
        "gemini",
    )
    cfg["llm_enrichment"]["provider"] = prov
    cfg["llm_enrichment"].setdefault("ollama", {})
    cfg["llm_enrichment"]["ollama"]["host"] = _string_setting(
        "OLLAMA_HOST",
        cfg["llm_enrichment"]["ollama"].get("host"),
        "http://localhost",
    )
    cfg["llm_enrichment"]["ollama"]["port"] = _int_setting(
        "OLLAMA_PORT",
        cfg["llm_enrichment"]["ollama"].get("port"),
        11434,
    )
    cfg["llm_enrichment"]["ollama"]["model"] = _string_setting(
        "OLLAMA_MODEL",
        cfg["llm_enrichment"]["ollama"].get("model"),
        "gemma:2b",
    )
    cfg["llm_enrichment"].setdefault("gemini", {})
    cfg["llm_enrichment"]["gemini"]["model"] = _string_setting(
        "GEMINI_MODEL",
        cfg["llm_enrichment"]["gemini"].get("model"),
        "gemini-1.5-flash",
    )

    # Prefer explicit cache envs. Otherwise pick a writable default.
    def _ensure_dir(p: str) -> str | None:
        try:
            os.makedirs(p, exist_ok=True)
            return p
        except Exception:
            return None

    # If user set any of these, respect them.
    hf_home = _clean_string(os.getenv("HF_HOME"))
    tf_cache = _clean_string(os.getenv("TRANSFORMERS_CACHE"))
    st_home = _clean_string(os.getenv("SENTENCE_TRANSFORMERS_HOME"))
    torch_home = _clean_string(os.getenv("TORCH_HOME"))

    # Choose a sane, writable default root
    default_root = _clean_string(os.getenv("MODEL_CACHE_DIR")) or os.path.join(
        os.getcwd(),
        ".models",
    )
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
        if not _clean_string(os.getenv(k)):
            os.environ[k] = v
        else:
            os.environ[k] = os.environ[k].strip()

    logger.info(f"Using device: {cfg['general']['device']}")
    return cfg


CONFIG = load_config()
