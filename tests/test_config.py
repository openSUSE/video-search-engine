import importlib
import sys

import pytest


@pytest.fixture(autouse=True)
def restore_config_module():
    original_module = sys.modules.get("core.config")
    yield
    sys.modules.pop("core.config", None)
    if original_module is not None:
        sys.modules["core.config"] = original_module


def _load_config_module(monkeypatch, tmp_path, config_text):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("ML_DEVICE", "cpu")
    sys.modules.pop("core.config", None)
    return importlib.import_module("core.config")


def test_hf_token_comes_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_from_environment")
    config_module = _load_config_module(
        monkeypatch,
        tmp_path,
        """
general:
  hf_token: "hf_from_yaml"
""",
    )

    assert config_module.CONFIG["general"]["hf_token"] == "hf_from_environment"


def test_hf_token_does_not_fall_back_to_yaml(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config_module = _load_config_module(
        monkeypatch,
        tmp_path,
        """
general:
  hf_token: "hf_from_yaml"
""",
    )

    assert config_module.CONFIG["general"]["hf_token"] is None


def test_environment_overrides_are_stripped(monkeypatch, tmp_path):
    monkeypatch.setenv("UI_HOST", " 0.0.0.0 ")
    monkeypatch.setenv("UI_PORT", " 5055 ")
    monkeypatch.setenv("API_HOST", " api ")
    monkeypatch.setenv("API_PORT", " 9000 ")
    monkeypatch.setenv("CHROMA_HOST", " chroma ")
    monkeypatch.setenv("CHROMA_PORT", " 8001 ")
    monkeypatch.setenv("CHROMA_COLLECTION", " video_segments ")
    monkeypatch.setenv("OUTPUT_DIR", " data/custom ")
    monkeypatch.setenv("LLM_PROVIDER", " ollama ")
    monkeypatch.setenv("OLLAMA_HOST", " http://ollama ")
    monkeypatch.setenv("OLLAMA_PORT", " 11435 ")
    monkeypatch.setenv("OLLAMA_MODEL", " gemma3 ")
    monkeypatch.setenv("GEMINI_MODEL", " gemini-flash ")
    config_module = _load_config_module(
        monkeypatch,
        tmp_path,
        """
general:
  default_output_dir: "data/processed"
ui:
  host: "127.0.0.1"
  port: 5050
api_server:
  host: "127.0.0.1"
  port: 1234
database:
  host: "localhost"
  port: 8000
  collection_name: "video_search_engine"
llm_enrichment:
  provider: "gemini"
  ollama:
    host: "http://localhost"
    port: 11434
    model: "gemma:2b"
  gemini:
    model: "gemini-1.5-flash"
""",
    )

    config = config_module.CONFIG
    assert config["general"]["default_output_dir"] == "data/custom"
    assert config["ui"] == {"host": "0.0.0.0", "port": 5055}
    assert config["api_server"] == {"host": "api", "port": 9000}
    assert config["database"] == {
        "host": "chroma",
        "port": 8001,
        "collection_name": "video_segments",
    }
    assert config["llm_enrichment"]["provider"] == "ollama"
    assert config["llm_enrichment"]["ollama"] == {
        "host": "http://ollama",
        "port": 11435,
        "model": "gemma3",
    }
    assert config["llm_enrichment"]["gemini"]["model"] == "gemini-flash"


def test_blank_environment_overrides_fall_back_to_config_values(monkeypatch, tmp_path):
    for env_name in [
        "UI_HOST",
        "UI_PORT",
        "API_HOST",
        "API_PORT",
        "CHROMA_HOST",
        "CHROMA_PORT",
        "CHROMA_COLLECTION",
        "OUTPUT_DIR",
        "LLM_PROVIDER",
        "OLLAMA_HOST",
        "OLLAMA_PORT",
        "OLLAMA_MODEL",
        "GEMINI_MODEL",
        "HF_TOKEN",
    ]:
        monkeypatch.setenv(env_name, " ")

    config_module = _load_config_module(
        monkeypatch,
        tmp_path,
        """
general:
  default_output_dir: "data/from-config"
  hf_token: "hf_from_yaml"
ui:
  host: "127.0.0.2"
  port: "5051"
api_server:
  host: "127.0.0.3"
  port: "1235"
database:
  host: "db"
  port: "8001"
  collection_name: "configured_collection"
llm_enrichment:
  provider: "configured_provider"
  ollama:
    host: "http://configured-ollama"
    port: "11435"
    model: "configured-ollama-model"
  gemini:
    model: "configured-gemini-model"
""",
    )

    config = config_module.CONFIG
    assert config["general"]["default_output_dir"] == "data/from-config"
    assert config["general"]["hf_token"] is None
    assert config["ui"] == {"host": "127.0.0.2", "port": 5051}
    assert config["api_server"] == {"host": "127.0.0.3", "port": 1235}
    assert config["database"] == {
        "host": "db",
        "port": 8001,
        "collection_name": "configured_collection",
    }
    assert config["llm_enrichment"]["provider"] == "configured_provider"
    assert config["llm_enrichment"]["ollama"] == {
        "host": "http://configured-ollama",
        "port": 11435,
        "model": "configured-ollama-model",
    }
    assert config["llm_enrichment"]["gemini"]["model"] == "configured-gemini-model"


@pytest.mark.parametrize(
    "env_name",
    ["UI_PORT", "API_PORT", "CHROMA_PORT", "OLLAMA_PORT"],
)
@pytest.mark.parametrize("raw_port", ["0", "-1", "65536", "not-a-port"])
def test_invalid_port_environment_overrides_fail_fast(
    monkeypatch,
    tmp_path,
    env_name,
    raw_port,
):
    monkeypatch.setenv(env_name, raw_port)

    with pytest.raises(ValueError, match=env_name):
        _load_config_module(monkeypatch, tmp_path, "{}")


@pytest.mark.parametrize(
    ("config_text", "message"),
    [
        ("ui:\n  port: 0\n", "UI_PORT"),
        ("api_server:\n  port: -1\n", "API_PORT"),
        ("database:\n  port: 65536\n", "CHROMA_PORT"),
        ("llm_enrichment:\n  ollama:\n    port: not-a-port\n", "OLLAMA_PORT"),
    ],
)
def test_invalid_configured_ports_fail_fast(
    monkeypatch,
    tmp_path,
    config_text,
    message,
):
    with pytest.raises(ValueError, match=message):
        _load_config_module(monkeypatch, tmp_path, config_text)


@pytest.mark.parametrize("config_text", ["[]", "not-a-mapping"])
def test_config_file_root_must_be_mapping(monkeypatch, tmp_path, config_text):
    with pytest.raises(ValueError, match="YAML mapping"):
        _load_config_module(monkeypatch, tmp_path, config_text)


@pytest.mark.parametrize("section_value", ["[]", "not-a-mapping"])
def test_config_sections_must_be_mappings(monkeypatch, tmp_path, section_value):
    with pytest.raises(ValueError, match="general"):
        _load_config_module(
            monkeypatch,
            tmp_path,
            f"""
general: {section_value}
""",
        )


@pytest.mark.parametrize(
    ("config_text", "message"),
    [
        ("llm_enrichment:\n  ollama: []\n", "llm_enrichment.ollama"),
        ("llm_enrichment:\n  gemini: not-a-mapping\n", "llm_enrichment.gemini"),
    ],
)
def test_nested_llm_config_sections_must_be_mappings(
    monkeypatch,
    tmp_path,
    config_text,
    message,
):
    with pytest.raises(ValueError, match=message):
        _load_config_module(monkeypatch, tmp_path, config_text)


def test_null_config_sections_are_treated_as_empty_mappings(monkeypatch, tmp_path):
    config_module = _load_config_module(
        monkeypatch,
        tmp_path,
        """
general:
""",
    )

    assert config_module.CONFIG["general"]["default_output_dir"] == "data/processed"
