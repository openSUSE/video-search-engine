import pytest

from app.ui import search_client


def _search_result(**updates):
    result = {
        "id": "segment-a",
        "score": 0.5,
        "start_time": 1,
        "end_time": 2.5,
        "title": "Opening",
        "summary": "A calm opening scene.",
        "video_filename": "demo.mp4",
        "speakers": "Alice",
    }
    result.update(updates)
    return result


class FakeSearchResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_search_api_url_uses_config_host_and_port():
    config = {"api_server": {"host": "api", "port": 1234}}

    assert search_client.search_api_url(config) == "http://api:1234/search"


def test_search_api_url_strips_config_host_and_port():
    config = {"api_server": {"host": "  api  ", "port": " 1234 "}}

    assert search_client.search_api_url(config) == "http://api:1234/search"


def test_search_api_url_uses_loopback_for_wildcard_config_host():
    config = {"api_server": {"host": "0.0.0.0", "port": 1234}}

    assert search_client.search_api_url(config) == "http://127.0.0.1:1234/search"


def test_search_api_url_uses_environment_when_config_is_not_provided(monkeypatch):
    monkeypatch.setenv("API_HOST", "api")
    monkeypatch.setenv("API_PORT", "1234")

    assert search_client.search_api_url() == "http://api:1234/search"


def test_search_api_url_strips_environment_values(monkeypatch):
    monkeypatch.setenv("API_HOST", "  api  ")
    monkeypatch.setenv("API_PORT", " 1234 ")

    assert search_client.search_api_url() == "http://api:1234/search"


def test_search_api_url_uses_loopback_for_wildcard_environment_host(monkeypatch):
    monkeypatch.setenv("API_HOST", "::")
    monkeypatch.setenv("API_PORT", "1234")

    assert search_client.search_api_url() == "http://127.0.0.1:1234/search"


def test_search_api_url_uses_defaults_for_blank_environment_values(monkeypatch):
    monkeypatch.setenv("API_HOST", " ")
    monkeypatch.setenv("API_PORT", "\t")

    assert search_client.search_api_url() == "http://localhost:1234/search"


def test_search_payload_includes_query_limit_and_video_filter():
    payload = search_client.search_payload("opening scene", "demo", top_k=7)

    assert payload == {
        "query": "opening scene",
        "top_k": 7,
        "video_filename": "demo",
    }


def test_search_payload_strips_query_and_video_filter():
    payload = search_client.search_payload("  opening scene  ", "  demo  ")

    assert payload == {
        "query": "opening scene",
        "top_k": 5,
        "video_filename": "demo",
    }


@pytest.mark.parametrize("video_filename", [None, "", "  "])
def test_search_payload_uses_null_for_omitted_video_filter(video_filename):
    payload = search_client.search_payload("opening scene", video_filename)

    assert payload == {
        "query": "opening scene",
        "top_k": 5,
        "video_filename": None,
    }


def test_request_exception_is_exposed_by_client_boundary():
    assert search_client.RequestException is search_client.requests.exceptions.RequestException


def test_post_search_uses_timeout_from_environment(monkeypatch):
    monkeypatch.setenv("SEARCH_API_TIMEOUT_SECONDS", "2.5")
    captured = {}
    expected_response = object()

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return expected_response

    monkeypatch.setattr(search_client.requests, "post", fake_post)

    response = search_client.post_search("http://api:1234/search", {"query": "demo"})

    assert response is expected_response
    assert captured == {
        "url": "http://api:1234/search",
        "json": {"query": "demo"},
        "timeout": 2.5,
    }


@pytest.mark.parametrize("raw_value", [None, "", "0", "-1", "abc", "nan", "inf"])
def test_search_timeout_falls_back_to_default_for_unusable_values(raw_value):
    assert (
        search_client.search_timeout_seconds(raw_value)
        == search_client.DEFAULT_SEARCH_TIMEOUT_SECONDS
    )


def test_search_results_from_response_validates_and_normalizes_results():
    response = FakeSearchResponse({"results": [_search_result()]})

    assert search_client.search_results_from_response(response) == [
        {
            "id": "segment-a",
            "score": 0.5,
            "start_time": 1.0,
            "end_time": 2.5,
            "title": "Opening",
            "summary": "A calm opening scene.",
            "video_filename": "demo.mp4",
            "speakers": "Alice",
        }
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (ValueError("bad json"), "valid JSON"),
        ([], "JSON object"),
        ({}, "results list"),
        ({"results": "not a list"}, "results list"),
        ({"results": ["not an object"]}, "result at index 0"),
        ({"results": [_search_result(title=123)]}, "title"),
        ({"results": [_search_result(score=-0.1)]}, "score"),
        ({"results": [_search_result(start_time=True)]}, "start_time"),
        ({"results": [_search_result(start_time=float("nan"))]}, "start_time"),
        ({"results": [_search_result(start_time=3, end_time=2)]}, "end_time"),
    ],
)
def test_search_results_from_response_rejects_malformed_payloads(payload, message):
    response = FakeSearchResponse(payload)

    with pytest.raises(ValueError, match=message):
        search_client.search_results_from_response(response)
