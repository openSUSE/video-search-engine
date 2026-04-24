import importlib
import sys


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
