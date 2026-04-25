import pytest

from app.ui import search_client


def test_search_api_url_uses_config_host_and_port():
    config = {"api_server": {"host": "api", "port": 1234}}

    assert search_client.search_api_url(config) == "http://api:1234/search"


def test_search_api_url_strips_config_host_and_port():
    config = {"api_server": {"host": "  api  ", "port": " 1234 "}}

    assert search_client.search_api_url(config) == "http://api:1234/search"


def test_search_api_url_uses_environment_when_config_is_not_provided(monkeypatch):
    monkeypatch.setenv("API_HOST", "api")
    monkeypatch.setenv("API_PORT", "1234")

    assert search_client.search_api_url() == "http://api:1234/search"


def test_search_api_url_strips_environment_values(monkeypatch):
    monkeypatch.setenv("API_HOST", "  api  ")
    monkeypatch.setenv("API_PORT", " 1234 ")

    assert search_client.search_api_url() == "http://api:1234/search"


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


@pytest.mark.parametrize("raw_value", [None, "", "0", "-1", "abc"])
def test_search_timeout_falls_back_to_default_for_unusable_values(raw_value):
    assert (
        search_client.search_timeout_seconds(raw_value)
        == search_client.DEFAULT_SEARCH_TIMEOUT_SECONDS
    )
