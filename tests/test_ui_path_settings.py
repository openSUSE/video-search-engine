from app.ui.path_settings import env_path_setting


def test_env_path_setting_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("VIDEO_DATA_PATH", raising=False)

    assert env_path_setting("VIDEO_DATA_PATH", "data/videos") == "data/videos"


def test_env_path_setting_strips_environment_value(monkeypatch):
    monkeypatch.setenv("VIDEO_DATA_PATH", "  /data/videos  ")

    assert env_path_setting("VIDEO_DATA_PATH", "data/videos") == "/data/videos"


def test_env_path_setting_uses_default_for_blank_environment_value(monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", " \t ")

    assert env_path_setting("OUTPUT_DIR", "data/processed") == "data/processed"
