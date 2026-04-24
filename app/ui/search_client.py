import os
from typing import Any, Mapping

import requests

DEFAULT_SEARCH_TIMEOUT_SECONDS = 10.0


def search_api_url(config: Mapping[str, Any]) -> str:
    api_config = config["api_server"]
    return f"http://{api_config['host']}:{api_config['port']}/search"


def search_payload(query: str, video_filename: str, top_k: int = 5) -> dict[str, Any]:
    return {
        "query": query,
        "top_k": top_k,
        "video_filename": video_filename,
    }


def search_timeout_seconds(raw_value: str | None = None) -> float:
    raw_timeout = os.getenv("SEARCH_API_TIMEOUT_SECONDS") if raw_value is None else raw_value
    if raw_timeout in (None, ""):
        return DEFAULT_SEARCH_TIMEOUT_SECONDS

    try:
        timeout = float(raw_timeout)
    except ValueError:
        return DEFAULT_SEARCH_TIMEOUT_SECONDS

    if timeout <= 0:
        return DEFAULT_SEARCH_TIMEOUT_SECONDS

    return timeout


def post_search(api_url: str, payload: Mapping[str, Any], timeout_seconds: float | None = None):
    timeout = search_timeout_seconds() if timeout_seconds is None else timeout_seconds
    return requests.post(api_url, json=dict(payload), timeout=timeout)
