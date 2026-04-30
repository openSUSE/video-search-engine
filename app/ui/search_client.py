import math
import os
from typing import Any, Mapping

import requests

from app.ui.url_settings import local_http_url

DEFAULT_SEARCH_TIMEOUT_SECONDS = 10.0
DEFAULT_API_HOST = "localhost"
DEFAULT_API_PORT = "1234"
RequestException = requests.exceptions.RequestException


def _response_string(
    result: Mapping[str, Any],
    field_name: str,
    result_index: int,
    *,
    allow_empty: bool = False,
) -> str:
    value = result.get(field_name)
    if not isinstance(value, str):
        raise ValueError(
            f"search result at index {result_index} must have string {field_name}"
        )
    value = value.strip()
    if not allow_empty and not value:
        raise ValueError(
            f"search result at index {result_index} must have non-empty string "
            f"{field_name}"
        )
    return value


def _response_number(
    result: Mapping[str, Any],
    field_name: str,
    result_index: int,
) -> float:
    value = result.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"search result at index {result_index} must have numeric {field_name}"
        )

    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(
            f"search result at index {result_index} must have usable {field_name}"
        )
    return number


def _search_result_from_payload(result: Any, result_index: int) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError(f"search result at index {result_index} must be a JSON object")

    start_time = _response_number(result, "start_time", result_index)
    end_time = _response_number(result, "end_time", result_index)
    if end_time < start_time:
        raise ValueError(
            f"search result at index {result_index} end_time must be "
            "greater than or equal to start_time"
        )

    return {
        "id": _response_string(result, "id", result_index),
        "score": _response_number(result, "score", result_index),
        "start_time": start_time,
        "end_time": end_time,
        "title": _response_string(result, "title", result_index),
        "summary": _response_string(result, "summary", result_index),
        "video_filename": _response_string(result, "video_filename", result_index),
        "speakers": _response_string(
            result,
            "speakers",
            result_index,
            allow_empty=True,
        ),
    }


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


def format_clock(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remainder = divmod(total_seconds, 60)
    return f"{minutes}m {remainder:02d}s"


def format_time_range(start_time: float, end_time: float) -> str:
    duration = max(0, int(end_time - start_time))
    return f"{format_clock(start_time)} → {format_clock(end_time)} ({duration}s)"


def search_results_from_response(response: Any) -> list[dict[str, Any]]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("Search API response must be valid JSON") from exc

    if not isinstance(payload, Mapping):
        raise ValueError("Search API response must be a JSON object")

    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Search API response must include a results list")

    return [
        _search_result_from_payload(result, result_index)
        for result_index, result in enumerate(results)
    ]
