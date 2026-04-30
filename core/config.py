# core/config.py
import ipaddress
import logging
import math
import os
from typing import Dict, Any
from urllib.parse import urlsplit

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
MIN_TCP_PORT = 1
MAX_TCP_PORT = 65535
DEFAULT_FILENAMES = {
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
    "enriched_segments": "final_enriched_segments.json",
}
DEFAULT_PARAMETERS = {
    "transcription": {"batch_size": 32},
    "audio": {"sample_rate": 16000},
    "audio_events": {"top_n": 3, "confidence_threshold": 0.1},
    "visual_captioning": {"max_new_tokens": 50},
    "action_recognition": {"num_frames": 16, "top_n": 3},
}
DEFAULT_MODELS = {
    "transcription": {
        "name": "base",
        "compute_type": "int8",
    },
    "audio_events": {
        "name": "MIT/ast-finetuned-audioset-10-10-0.4593",
    },
    "visual_captioning": {
        "name": "Salesforce/blip-image-captioning-base",
    },
    "embedding": {
        "name": "all-MiniLM-L6-v2",
    },
    "action_recognition": {
        "name": "MCG-NJU/videomae-base-finetuned-kinetics",
    },
}
LLM_PROVIDERS = {"gemini", "ollama"}


def _yaml(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                loaded_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"config file {path} must be valid YAML") from exc
        except OSError as exc:
            raise ValueError(
                f"config file {path} must be a readable YAML file"
            ) from exc
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


def _ensure_nested_config_section(
    parent: Dict[str, Any],
    section_name: str,
    dotted_name: str,
) -> Dict[str, Any]:
    section = parent.setdefault(section_name, {})
    if section is None:
        parent[section_name] = {}
        return parent[section_name]
    if not isinstance(section, dict):
        raise ValueError(f"config section '{dotted_name}' must be a mapping")
    return section


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def _coerced_setting(env_name: str, config_value: Any, default: str) -> str:
    return _clean_string(os.getenv(env_name)) or _clean_string(config_value) or default


def _string_setting(env_name: str, config_value: Any, default: str) -> str:
    env_value = _clean_string(os.getenv(env_name))
    if env_value is not None:
        return env_value

    if config_value is None:
        return default
    if not isinstance(config_value, str):
        raise ValueError(f"{env_name} must be a string")

    return _clean_string(config_value) or default


def _host_setting(env_name: str, config_value: Any, default: str) -> str:
    host = _string_setting(env_name, config_value, default)
    invalid_message = (
        f"{env_name} must be a hostname or IP address without scheme, path, or port"
    )

    if (
        "://" in host
        or any(character.isspace() for character in host)
        or any(character in host for character in "/\\?#@")
    ):
        raise ValueError(invalid_message)

    if host.startswith("[") or host.endswith("]"):
        if not (host.startswith("[") and host.endswith("]")):
            raise ValueError(invalid_message)
        try:
            ip_address = ipaddress.ip_address(host[1:-1])
        except ValueError as exc:
            raise ValueError(invalid_message) from exc
        if ip_address.version != 6:
            raise ValueError(invalid_message)
        return host

    if ":" in host:
        try:
            ip_address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError(invalid_message) from exc
        if ip_address.version != 6:
            raise ValueError(invalid_message)

    return host


def _plain_string_setting(dotted_name: str, config_value: Any, default: str) -> str:
    if config_value is None:
        return default
    if not isinstance(config_value, str):
        raise ValueError(f"{dotted_name} must be a string")

    return _clean_string(config_value) or default


def _choice_setting(
    env_name: str,
    config_value: Any,
    default: str,
    allowed_values: set[str],
) -> str:
    value = _string_setting(env_name, config_value, default).lower()
    if value not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(f"{env_name} must be one of: {allowed}")
    return value


def _http_origin_setting(env_name: str, config_value: Any, default: str) -> str:
    origin = _string_setting(env_name, config_value, default)
    invalid_message = (
        f"{env_name} must be an http(s) URL origin without path, port, "
        "credentials, query, or fragment"
    )
    try:
        parts = urlsplit(origin)
        embedded_port = parts.port
    except ValueError as exc:
        raise ValueError(invalid_message) from exc

    if (
        any(character.isspace() for character in origin)
        or "\\" in origin
        or parts.scheme.lower() not in {"http", "https"}
        or not parts.netloc
        or not parts.hostname
        or parts.netloc.endswith(":")
        or parts.username
        or parts.password
        or embedded_port is not None
        or parts.path not in {"", "/"}
        or parts.query
        or parts.fragment
    ):
        raise ValueError(invalid_message)

    return f"{parts.scheme.lower()}://{parts.netloc}"


def _positive_number_setting(
    dotted_name: str,
    config_value: Any,
    default: float,
) -> float | int:
    invalid_message = f"{dotted_name} must be a finite positive number"
    if config_value is None:
        return default
    if isinstance(config_value, bool) or not isinstance(config_value, (int, float)):
        raise ValueError(invalid_message)

    number = float(config_value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(invalid_message)
    return config_value


def _positive_int_setting(
    dotted_name: str,
    config_value: Any,
    default: int,
) -> int:
    invalid_message = f"{dotted_name} must be a positive integer"
    if config_value is None:
        return default
    if type(config_value) is not int or config_value <= 0:
        raise ValueError(invalid_message)
    return config_value


def _unit_interval_setting(
    dotted_name: str,
    config_value: Any,
    default: float,
) -> float | int:
    invalid_message = f"{dotted_name} must be a finite number between 0 and 1"
    if config_value is None:
        return default
    if isinstance(config_value, bool) or not isinstance(config_value, (int, float)):
        raise ValueError(invalid_message)

    number = float(config_value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(invalid_message)
    return config_value


def _port_setting(env_name: str, config_value: Any, default: int) -> int:
    raw_port = _coerced_setting(env_name, config_value, str(default))
    invalid_message = (
        f"{env_name} must be a TCP port between {MIN_TCP_PORT} and {MAX_TCP_PORT}"
    )
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError(invalid_message) from exc

    if not MIN_TCP_PORT <= port <= MAX_TCP_PORT:
        raise ValueError(invalid_message)
    return port


def _filename_setting(key: str, config_value: Any, default: str) -> str:
    invalid_message = f"filenames.{key} must be a non-empty filename"
    if config_value is None:
        return default
    if not isinstance(config_value, str):
        raise ValueError(invalid_message)

    filename = config_value.strip()
    if (
        not filename
        or filename in {".", ".."}
        or os.path.basename(filename) != filename
        or "\\" in filename
    ):
        raise ValueError(invalid_message)
    return filename


def _normalize_filenames(filenames_config: Dict[str, Any]) -> None:
    for key, default in DEFAULT_FILENAMES.items():
        filenames_config[key] = _filename_setting(
            key,
            filenames_config.get(key),
            default,
        )


def _normalize_models(models_config: Dict[str, Any]) -> None:
    transcription_config = _ensure_nested_config_section(
        models_config,
        "transcription",
        "models.transcription",
    )
    transcription_config["name"] = _plain_string_setting(
        "models.transcription.name",
        transcription_config.get("name"),
        DEFAULT_MODELS["transcription"]["name"],
    )
    transcription_config["compute_type"] = _plain_string_setting(
        "models.transcription.compute_type",
        transcription_config.get("compute_type"),
        DEFAULT_MODELS["transcription"]["compute_type"],
    )

    audio_events_config = _ensure_nested_config_section(
        models_config,
        "audio_events",
        "models.audio_events",
    )
    audio_events_config["name"] = _plain_string_setting(
        "models.audio_events.name",
        audio_events_config.get("name"),
        DEFAULT_MODELS["audio_events"]["name"],
    )

    visual_captioning_config = _ensure_nested_config_section(
        models_config,
        "visual_captioning",
        "models.visual_captioning",
    )
    visual_captioning_config["name"] = _plain_string_setting(
        "models.visual_captioning.name",
        visual_captioning_config.get("name"),
        DEFAULT_MODELS["visual_captioning"]["name"],
    )

    embedding_config = _ensure_nested_config_section(
        models_config,
        "embedding",
        "models.embedding",
    )
    embedding_config["name"] = _plain_string_setting(
        "models.embedding.name",
        embedding_config.get("name"),
        DEFAULT_MODELS["embedding"]["name"],
    )

    action_recognition_config = _ensure_nested_config_section(
        models_config,
        "action_recognition",
        "models.action_recognition",
    )
    action_recognition_config["name"] = _plain_string_setting(
        "models.action_recognition.name",
        action_recognition_config.get("name"),
        DEFAULT_MODELS["action_recognition"]["name"],
    )


def _normalize_parameters(parameters_config: Dict[str, Any]) -> None:
    transcription_config = _ensure_nested_config_section(
        parameters_config,
        "transcription",
        "parameters.transcription",
    )
    transcription_config["batch_size"] = _positive_int_setting(
        "parameters.transcription.batch_size",
        transcription_config.get("batch_size"),
        DEFAULT_PARAMETERS["transcription"]["batch_size"],
    )

    audio_config = _ensure_nested_config_section(
        parameters_config,
        "audio",
        "parameters.audio",
    )
    audio_config["sample_rate"] = _positive_int_setting(
        "parameters.audio.sample_rate",
        audio_config.get("sample_rate"),
        DEFAULT_PARAMETERS["audio"]["sample_rate"],
    )

    audio_events_config = _ensure_nested_config_section(
        parameters_config,
        "audio_events",
        "parameters.audio_events",
    )
    audio_events_config["top_n"] = _positive_int_setting(
        "parameters.audio_events.top_n",
        audio_events_config.get("top_n"),
        DEFAULT_PARAMETERS["audio_events"]["top_n"],
    )
    audio_events_config["confidence_threshold"] = _unit_interval_setting(
        "parameters.audio_events.confidence_threshold",
        audio_events_config.get("confidence_threshold"),
        DEFAULT_PARAMETERS["audio_events"]["confidence_threshold"],
    )

    visual_captioning_config = _ensure_nested_config_section(
        parameters_config,
        "visual_captioning",
        "parameters.visual_captioning",
    )
    visual_captioning_config["max_new_tokens"] = _positive_int_setting(
        "parameters.visual_captioning.max_new_tokens",
        visual_captioning_config.get("max_new_tokens"),
        DEFAULT_PARAMETERS["visual_captioning"]["max_new_tokens"],
    )

    action_recognition_config = _ensure_nested_config_section(
        parameters_config,
        "action_recognition",
        "parameters.action_recognition",
    )
    action_recognition_config["num_frames"] = _positive_int_setting(
        "parameters.action_recognition.num_frames",
        action_recognition_config.get("num_frames"),
        DEFAULT_PARAMETERS["action_recognition"]["num_frames"],
    )
    action_recognition_config["top_n"] = _positive_int_setting(
        "parameters.action_recognition.top_n",
        action_recognition_config.get("top_n"),
        DEFAULT_PARAMETERS["action_recognition"]["top_n"],
    )


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
    cfg["ui"]["host"] = _host_setting("UI_HOST", cfg["ui"].get("host"), "0.0.0.0")
    cfg["ui"]["port"] = _port_setting("UI_PORT", cfg["ui"].get("port"), 5050)
    cfg["api_server"]["host"] = _host_setting(
        "API_HOST",
        cfg["api_server"].get("host"),
        "0.0.0.0",
    )
    cfg["api_server"]["port"] = _port_setting(
        "API_PORT",
        cfg["api_server"].get("port"),
        1234,
    )

    # ------- database -------
    cfg["database"]["host"] = _host_setting(
        "CHROMA_HOST",
        cfg["database"].get("host"),
        "localhost",
    )
    cfg["database"]["port"] = _port_setting(
        "CHROMA_PORT",
        cfg["database"].get("port"),
        8000,
    )
    cfg["database"]["collection_name"] = _string_setting(
        "CHROMA_COLLECTION",
        cfg["database"].get("collection_name"),
        "video_search_engine",
    )

    # ------- artifact filenames -------
    _normalize_filenames(cfg["filenames"])

    # ------- model identifiers -------
    _normalize_models(cfg["models"])

    # ------- extraction/runtime parameters -------
    _normalize_parameters(cfg["parameters"])

    # ------- LLM provider overrides -------
    prov = _choice_setting(
        "LLM_PROVIDER",
        cfg["llm_enrichment"].get("provider"),
        "gemini",
        LLM_PROVIDERS,
    )
    cfg["llm_enrichment"]["provider"] = prov
    ollama_config = _ensure_nested_config_section(
        cfg["llm_enrichment"],
        "ollama",
        "llm_enrichment.ollama",
    )
    ollama_config["host"] = _http_origin_setting(
        "OLLAMA_HOST",
        ollama_config.get("host"),
        "http://localhost",
    )
    ollama_config["port"] = _port_setting(
        "OLLAMA_PORT",
        ollama_config.get("port"),
        11434,
    )
    ollama_config["model"] = _string_setting(
        "OLLAMA_MODEL",
        ollama_config.get("model"),
        "gemma:2b",
    )
    ollama_config["timeout_sec"] = _positive_number_setting(
        "llm_enrichment.ollama.timeout_sec",
        ollama_config.get("timeout_sec"),
        120,
    )
    gemini_config = _ensure_nested_config_section(
        cfg["llm_enrichment"],
        "gemini",
        "llm_enrichment.gemini",
    )
    gemini_config["model"] = _string_setting(
        "GEMINI_MODEL",
        gemini_config.get("model"),
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

    logger.info("Using device: %s", cfg["general"]["device"])
    return cfg


CONFIG = load_config()
