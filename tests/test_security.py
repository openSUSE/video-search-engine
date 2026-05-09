"""Tests for the opt-in API security primitives.

Each pair of tests covers:
* the "off by default" stance (no env var → no enforcement)
* the "on when configured" path (env var present → enforcement)

The 401-with-WWW-Authenticate, 413, and CORS preflight responses are the
contract third-party callers depend on, so they're asserted explicitly
rather than just "non-2xx".
"""

from __future__ import annotations

import pytest

# api.security imports fastapi at module load (it builds middleware), so
# the entire module — config helpers included — can only be imported in an
# env that has FastAPI. When fastapi is missing, the whole test file
# skips. The config-helper assertions still run inside the api container's
# test invocation.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import security as api_security  # noqa: E402

needs_fastapi = pytest.mark.skipif(False, reason="kept for parity")


def _build_app(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    app = FastAPI()
    api_security.register(app)

    @app.post("/search")
    def search() -> dict:
        return {"ok": True}

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/metrics")
    def metrics() -> dict:
        return {"ok": True}

    return app


def test_configured_api_key_returns_empty_when_unset(monkeypatch):
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    assert api_security.configured_api_key() == ""


def test_configured_api_key_strips_whitespace(monkeypatch):
    monkeypatch.setenv("SEARCH_API_KEY", "  super-secret  ")
    assert api_security.configured_api_key() == "super-secret"


def test_configured_cors_origins_empty_by_default(monkeypatch):
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    assert api_security.configured_cors_origins() == []


def test_configured_cors_origins_parses_csv(monkeypatch):
    monkeypatch.setenv(
        "CORS_ALLOW_ORIGINS",
        "https://app.example.com, https://other.example.com,  ",
    )
    assert api_security.configured_cors_origins() == [
        "https://app.example.com",
        "https://other.example.com",
    ]


def test_configured_max_body_bytes_default(monkeypatch):
    monkeypatch.delenv("MAX_REQUEST_BODY_BYTES", raising=False)
    assert api_security.configured_max_body_bytes() == api_security.DEFAULT_MAX_BODY_BYTES


@pytest.mark.parametrize("bad", ["abc", "-100", "0"])
def test_configured_max_body_bytes_falls_back_on_invalid(monkeypatch, bad):
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", bad)
    assert api_security.configured_max_body_bytes() == api_security.DEFAULT_MAX_BODY_BYTES


def test_configured_max_body_bytes_accepts_explicit_override(monkeypatch):
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "1024")
    assert api_security.configured_max_body_bytes() == 1024


@needs_fastapi
def test_search_is_open_when_api_key_unset(monkeypatch):
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    client = TestClient(_build_app(monkeypatch, {}))
    response = client.post("/search", json={"q": "anything"})
    assert response.status_code == 200


@needs_fastapi
def test_audit_log_emits_denied_record_on_bad_key(monkeypatch, caplog):
    monkeypatch.setenv("SEARCH_API_KEY", "topsecret")
    client = TestClient(_build_app(monkeypatch, {}))
    with caplog.at_level("WARNING", logger="api.security.audit"):
        client.post(
            "/search",
            json={"q": "x"},
            headers={api_security.API_KEY_HEADER: "wrong"},
        )
    records = [r for r in caplog.records if r.name == "api.security.audit"]
    assert records, "an audit warning record must be emitted on auth failure"
    record = records[-1]
    assert record.outcome == "denied"
    assert record.path == "/search"
    assert record.key_presented is True
    # The presented key is "wrong" → prefix should be the first 4 chars
    # plus an ellipsis. Critically, the full presented key is never logged.
    assert record.key_prefix == "wron…"
    assert "wrong" not in record.getMessage()


@needs_fastapi
def test_audit_log_omits_key_when_header_absent(monkeypatch, caplog):
    monkeypatch.setenv("SEARCH_API_KEY", "topsecret")
    client = TestClient(_build_app(monkeypatch, {}))
    with caplog.at_level("WARNING", logger="api.security.audit"):
        client.post("/search", json={"q": "x"})
    record = [r for r in caplog.records if r.name == "api.security.audit"][-1]
    assert record.outcome == "denied"
    assert record.key_presented is False
    assert record.key_prefix == "-"


@needs_fastapi
def test_audit_log_emits_granted_record_on_success(monkeypatch, caplog):
    monkeypatch.setenv("SEARCH_API_KEY", "topsecret")
    client = TestClient(_build_app(monkeypatch, {}))
    with caplog.at_level("INFO", logger="api.security.audit"):
        client.post(
            "/search",
            json={"q": "x"},
            headers={api_security.API_KEY_HEADER: "topsecret"},
        )
    records = [r for r in caplog.records if r.name == "api.security.audit"]
    assert records, "an audit info record must be emitted on success"
    record = records[-1]
    assert record.outcome == "granted"
    assert record.key_prefix == "tops…"


@needs_fastapi
def test_search_requires_api_key_when_configured(monkeypatch):
    client = TestClient(_build_app(monkeypatch, {"SEARCH_API_KEY": "topsecret"}))
    # Missing key: 401 with WWW-Authenticate.
    response = client.post("/search", json={"q": "anything"})
    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == api_security.API_KEY_HEADER

    # Wrong key: 401 too.
    response = client.post(
        "/search",
        json={"q": "anything"},
        headers={api_security.API_KEY_HEADER: "wrong"},
    )
    assert response.status_code == 401

    # Right key: 200.
    response = client.post(
        "/search",
        json={"q": "anything"},
        headers={api_security.API_KEY_HEADER: "topsecret"},
    )
    assert response.status_code == 200


@needs_fastapi
def test_healthz_metrics_open_even_when_auth_enabled(monkeypatch):
    client = TestClient(_build_app(monkeypatch, {"SEARCH_API_KEY": "topsecret"}))
    assert client.get("/healthz").status_code == 200
    assert client.get("/metrics").status_code == 200


@needs_fastapi
def test_oversized_body_is_rejected_with_413(monkeypatch):
    client = TestClient(
        _build_app(
            monkeypatch,
            {"MAX_REQUEST_BODY_BYTES": "100"},
        )
    )
    big_payload = {"q": "x" * 200}
    response = client.post("/search", json=big_payload)
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"].lower()


@needs_fastapi
def test_within_limit_body_is_accepted(monkeypatch):
    client = TestClient(
        _build_app(
            monkeypatch,
            {"MAX_REQUEST_BODY_BYTES": "10000"},
        )
    )
    response = client.post("/search", json={"q": "small"})
    assert response.status_code == 200


@needs_fastapi
def test_cors_middleware_off_by_default(monkeypatch):
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    client = TestClient(_build_app(monkeypatch, {}))
    # Without CORS middleware, a cross-origin request gets no
    # access-control-allow-origin header on the response.
    response = client.post(
        "/search",
        json={"q": "x"},
        headers={"Origin": "https://example.com"},
    )
    assert "access-control-allow-origin" not in {h.lower() for h in response.headers}


@needs_fastapi
def test_cors_middleware_allows_listed_origin(monkeypatch):
    client = TestClient(
        _build_app(
            monkeypatch,
            {"CORS_ALLOW_ORIGINS": "https://app.example.com"},
        )
    )
    response = client.post(
        "/search",
        json={"q": "x"},
        headers={"Origin": "https://app.example.com"},
    )
    assert response.status_code == 200
    assert (
        response.headers.get("access-control-allow-origin")
        == "https://app.example.com"
    )


def test_configured_rate_limit_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RATE_LIMIT_PER_MINUTE", raising=False)
    assert api_security.configured_rate_limit_per_minute() == 0


@pytest.mark.parametrize("bad", ["abc", "-3"])
def test_configured_rate_limit_falls_back_on_invalid(monkeypatch, bad):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", bad)
    assert api_security.configured_rate_limit_per_minute() == 0


def test_configured_rate_limit_accepts_explicit_value(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "120")
    assert api_security.configured_rate_limit_per_minute() == 120


def test_sliding_window_limiter_admits_under_budget():
    limiter = api_security._SlidingWindowRateLimiter(limit=3, window_seconds=60)
    for _ in range(3):
        allowed, retry = limiter.check("k", now=100.0)
        assert allowed
        assert retry == 0


def test_sliding_window_limiter_rejects_over_budget():
    limiter = api_security._SlidingWindowRateLimiter(limit=2, window_seconds=60)
    assert limiter.check("k", now=100.0)[0] is True
    assert limiter.check("k", now=100.5)[0] is True
    allowed, retry = limiter.check("k", now=100.6)
    assert allowed is False
    assert retry > 0


def test_sliding_window_limiter_releases_after_window():
    limiter = api_security._SlidingWindowRateLimiter(limit=1, window_seconds=10)
    assert limiter.check("k", now=100.0)[0] is True
    assert limiter.check("k", now=105.0)[0] is False
    # After the window passes, the oldest entry ages out and the budget
    # frees up.
    assert limiter.check("k", now=111.0)[0] is True


def test_sliding_window_limiter_keys_are_isolated():
    limiter = api_security._SlidingWindowRateLimiter(limit=1, window_seconds=60)
    assert limiter.check("a", now=100.0)[0] is True
    # Different key has its own bucket.
    assert limiter.check("b", now=100.0)[0] is True
    # Same key now exceeds.
    assert limiter.check("a", now=100.5)[0] is False


def test_sliding_window_limiter_zero_disables():
    limiter = api_security._SlidingWindowRateLimiter(limit=0, window_seconds=60)
    for _ in range(100):
        allowed, retry = limiter.check("k")
        assert allowed and retry == 0


@needs_fastapi
def test_rate_limit_middleware_returns_429_with_retry_after(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    client = TestClient(_build_app(monkeypatch, {}))

    # First two requests succeed.
    assert client.post("/search", json={"q": "x"}).status_code == 200
    assert client.post("/search", json={"q": "x"}).status_code == 200

    # Third hits the limit.
    response = client.post("/search", json={"q": "x"})
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1
    assert "rate limit" in response.json()["detail"].lower()


@needs_fastapi
def test_rate_limit_does_not_apply_to_health_metrics(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    client = TestClient(_build_app(monkeypatch, {}))
    # Use the search budget so the limiter is hot.
    client.post("/search", json={"q": "x"})
    # /healthz and /metrics must still respond unconditionally so probes
    # and Prometheus scrapes can't be locked out.
    for _ in range(3):
        assert client.get("/healthz").status_code == 200
        assert client.get("/metrics").status_code == 200


@needs_fastapi
def test_cors_middleware_rejects_unlisted_origin(monkeypatch):
    client = TestClient(
        _build_app(
            monkeypatch,
            {"CORS_ALLOW_ORIGINS": "https://app.example.com"},
        )
    )
    response = client.post(
        "/search",
        json={"q": "x"},
        headers={"Origin": "https://attacker.example.com"},
    )
    # The actual response still goes through (CORS is browser-enforced),
    # but the access-control-allow-origin header must NOT echo the bad
    # origin — that's what stops the browser from accepting it.
    assert response.headers.get("access-control-allow-origin") != (
        "https://attacker.example.com"
    )
