import math
import os
from typing import Any, Mapping

import requests

from app.ui.url_settings import local_http_url

DEFAULT_SEARCH_TIMEOUT_SECONDS = 10.0
DEFAULT_API_HOST = "localhost"
DEFAULT_API_PORT = "1234"
RequestException = requests.exceptions.RequestException


def _url_component(value: Any, default: str) -> str:
    if value is None:
        return default

    value = str(value).strip()
    return value or default


def _payload_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _optional_payload_text(value: Any) -> str | None:
    normalized_value = _payload_text(value)
    return normalized_value or None


def search_api_url(config: Mapping[str, Any] | None = None) -> str:
    if config is None:
        host = _url_component(os.getenv("API_HOST"), DEFAULT_API_HOST)
        port = _url_component(os.getenv("API_PORT"), DEFAULT_API_PORT)
    else:
        api_config = config["api_server"]
        host = _url_component(api_config.get("host"), DEFAULT_API_HOST)
        port = _url_component(api_config.get("port"), DEFAULT_API_PORT)

    return f"{local_http_url(host, port)}/search"


def search_payload(
    query: str,
    video_filename: str | None,
    top_k: int = 5,
) -> dict[str, Any]:
    return {
        "query": _payload_text(query),
        "top_k": top_k,
        "video_filename": _optional_payload_text(video_filename),
    }


def search_timeout_seconds(raw_value: str | None = None) -> float:
    raw_timeout = os.getenv("SEARCH_API_TIMEOUT_SECONDS") if raw_value is None else raw_value
    if raw_timeout is None:
        return DEFAULT_SEARCH_TIMEOUT_SECONDS

    raw_timeout = str(raw_timeout).strip()
    if not raw_timeout:
        return DEFAULT_SEARCH_TIMEOUT_SECONDS

    try:
        timeout = float(raw_timeout)
    except ValueError:
        return DEFAULT_SEARCH_TIMEOUT_SECONDS

    if not math.isfinite(timeout) or timeout <= 0:
        return DEFAULT_SEARCH_TIMEOUT_SECONDS

    return timeout


def post_search(api_url: str, payload: Mapping[str, Any], timeout_seconds: float | None = None):
    timeout = search_timeout_seconds() if timeout_seconds is None else timeout_seconds
    return requests.post(api_url, json=dict(payload), timeout=timeout)
