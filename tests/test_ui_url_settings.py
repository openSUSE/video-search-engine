import pytest

from app.ui.url_settings import local_http_url


def test_local_http_url_replaces_wildcard_bind_hosts_with_loopback():
    assert local_http_url("0.0.0.0", 5050) == "http://127.0.0.1:5050"
    assert local_http_url("::", "5050") == "http://127.0.0.1:5050"


def test_local_http_url_preserves_specific_hosts():
    assert local_http_url(" localhost ", " 8501 ") == "http://localhost:8501"


def test_local_http_url_brackets_ipv6_literals():
    assert local_http_url("::1", 5050) == "http://[::1]:5050"
    assert local_http_url("[::1]", 5050) == "http://[::1]:5050"


@pytest.mark.parametrize("port", [None, "", " ", "abc", "0", "-1", "65536", "1.5"])
def test_local_http_url_rejects_invalid_ports(port):
    with pytest.raises(ValueError, match="TCP port"):
        local_http_url("localhost", port)


@pytest.mark.parametrize(
    "host",
    [
        "http://localhost",
        "localhost:5050",
        "api/search",
        "user@api",
        "bad host",
        "[localhost]",
        "[::1",
    ],
)
def test_local_http_url_rejects_malformed_hosts(host):
    with pytest.raises(ValueError, match="hostname or IP address"):
        local_http_url(host, 5050)
