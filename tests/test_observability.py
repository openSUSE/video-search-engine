"""Tests for the observability primitives.

Covers:
* contextvar request id binding/clearing
* JSON formatter shape and request-id surfacing
* request-id middleware: incoming header is honored, missing header gets a
  fresh id, response always carries the header
* /metrics endpoint smoke (skips when prometheus_client isn't installed)
"""

from __future__ import annotations

import json
import logging

import pytest

from core import logger as core_logger
from core.observability import (
    REQUEST_ID_HEADER,
    current_request_id,
    new_request_id,
    request_id_context,
)


# The HTTP middleware tests need FastAPI + httpx (TestClient transport);
# the contextvar / formatter tests don't. The skip annotation is per-test
# below so the unit-only suite still validates the contextvar+logger path
# even in environments without FastAPI installed.
_fastapi_available = True
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api import observability as api_observability
except Exception:  # pragma: no cover - exercised when fastapi is absent
    _fastapi_available = False

needs_fastapi = pytest.mark.skipif(
    not _fastapi_available,
    reason="FastAPI / httpx not installed (requirements-api).",
)


def test_request_id_context_binds_and_clears():
    assert current_request_id() is None
    with request_id_context("abc123"):
        assert current_request_id() == "abc123"
    assert current_request_id() is None


def test_request_id_context_resets_on_exception():
    assert current_request_id() is None
    with pytest.raises(RuntimeError):
        with request_id_context("rid"):
            assert current_request_id() == "rid"
            raise RuntimeError("boom")
    assert current_request_id() is None


def test_new_request_id_is_unique_hex():
    seen = {new_request_id() for _ in range(50)}
    assert len(seen) == 50
    for rid in seen:
        assert len(rid) == 12
        int(rid, 16)  # raises ValueError if non-hex


def _capture_log_record(formatter: logging.Formatter, *, request_id: str | None) -> str:
    """Format a single record through ``formatter`` while ``request_id`` is bound."""
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    # Mimic what the project filter does; the formatter itself doesn't read
    # the contextvar — the filter attaches it onto the record.
    record.request_id = request_id or "-"
    return formatter.format(record)


def test_json_formatter_includes_request_id_and_message():
    formatter = core_logger._JsonFormatter()
    output = _capture_log_record(formatter, request_id="rid-1")
    payload = json.loads(output)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["message"] == "hello world"
    assert payload["request_id"] == "rid-1"
    assert "timestamp" in payload


def test_json_formatter_renders_dash_when_no_request_id():
    formatter = core_logger._JsonFormatter()
    output = _capture_log_record(formatter, request_id=None)
    payload = json.loads(output)
    assert payload["request_id"] == "-"


def test_request_id_filter_pulls_from_contextvar():
    flt = core_logger._RequestIdFilter()
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="m",
        args=(),
        exc_info=None,
    )
    with request_id_context("ctx-rid"):
        assert flt.filter(record) is True
        assert record.request_id == "ctx-rid"


def test_request_id_filter_falls_back_to_dash():
    flt = core_logger._RequestIdFilter()
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="m",
        args=(),
        exc_info=None,
    )
    flt.filter(record)
    assert record.request_id == "-"


def _build_app_with_observability():
    app = FastAPI()
    api_observability.register(app)

    @app.get("/echo")
    def echo() -> dict:
        return {"request_id": current_request_id()}

    return app


@needs_fastapi
def test_middleware_mints_request_id_when_header_absent():
    client = TestClient(_build_app_with_observability())
    response = client.get("/echo")
    assert response.status_code == 200
    rid = response.headers.get(REQUEST_ID_HEADER)
    assert rid and len(rid) == 12
    body = response.json()
    assert body["request_id"] == rid


@needs_fastapi
def test_middleware_propagates_incoming_request_id_header():
    client = TestClient(_build_app_with_observability())
    response = client.get("/echo", headers={REQUEST_ID_HEADER: "explicit-id-123"})
    assert response.status_code == 200
    assert response.headers.get(REQUEST_ID_HEADER) == "explicit-id-123"
    assert response.json()["request_id"] == "explicit-id-123"


@needs_fastapi
def test_metrics_endpoint_exposes_prometheus_payload():
    pytest.importorskip("prometheus_client")
    app = _build_app_with_observability()

    @app.get("/sample")
    def sample() -> dict:
        return {}

    client = TestClient(app)
    # Drive a couple requests so counters move off zero.
    client.get("/sample")
    client.get("/sample")
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    body = metrics.text
    # Counter and histogram both register; Prometheus format always emits
    # HELP and TYPE lines for declared metric families.
    assert "video_se_http_requests_total" in body
    assert "video_se_http_request_duration_seconds" in body


@needs_fastapi
def test_metrics_endpoint_absent_when_prometheus_missing(monkeypatch):
    """Simulate prometheus_client not being importable."""
    import importlib
    import sys

    real_mod = sys.modules.pop("prometheus_client", None)
    monkeypatch.setitem(sys.modules, "prometheus_client", None)

    # Force re-import of api.observability so its lazy import path is exercised.
    sys.modules.pop("api.observability", None)
    try:
        reloaded = importlib.import_module("api.observability")
        app = FastAPI()
        reloaded.register(app)
        client = TestClient(app)
        response = client.get("/metrics")
        # Without prometheus_client the route isn't registered → 404.
        assert response.status_code == 404
    finally:
        sys.modules.pop("api.observability", None)
        if real_mod is not None:
            sys.modules["prometheus_client"] = real_mod
