import os
from typing import Any


def _clean_path(value: Any) -> str | None:
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def env_path_setting(env_name: str, default: str) -> str:
    return _clean_path(os.getenv(env_name)) or _clean_path(default) or ""
