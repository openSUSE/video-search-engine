"""Tests for ``core.hf_preflight``.

The preflight makes one HEAD request per gated model. We stub
``requests.head`` so tests don't depend on network or HF API behavior.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest

from core import hf_preflight
from core.hf_preflight import (
    GATED_MODELS,
    ModelAccessResult,
    PreflightReport,
    check_huggingface_access,
    log_preflight_report,
)


class FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


@pytest.fixture
def patched_requests(monkeypatch):
    """Capture every requests.head call so each test can pin behavior."""
    calls = []

    def factory(status_code: int):
        def head(url, headers=None, timeout=None, allow_redirects=False):
            calls.append((url, headers, timeout))
            return FakeResponse(status_code)

        return head

    return calls, factory


def test_missing_token_returns_missing_token_status_for_each_model():
    report = check_huggingface_access(token="")
    assert len(report.models) == len(GATED_MODELS)
    assert all(r.status == "missing_token" for r in report.models)
    assert report.all_ok is False


def test_token_valid_and_terms_accepted_returns_ok(patched_requests, monkeypatch):
    calls, factory = patched_requests
    import requests

    monkeypatch.setattr(requests, "head", factory(200))
    report = check_huggingface_access(token="hf_real")
    assert all(r.status == "ok" for r in report.models)
    assert report.all_ok is True
    # We sent one request per gated model with the bearer token.
    assert len(calls) == len(GATED_MODELS)
    for url, headers, _ in calls:
        assert headers.get("Authorization") == "Bearer hf_real"
        assert "huggingface.co" in url


def test_403_returns_needs_terms(patched_requests, monkeypatch):
    _calls, factory = patched_requests
    import requests

    monkeypatch.setattr(requests, "head", factory(403))
    report = check_huggingface_access(token="hf_token_no_access")
    assert all(r.status == "needs_terms" for r in report.models)
    assert all("hf.co" in r.detail for r in report.models)


def test_401_returns_invalid_token(patched_requests, monkeypatch):
    _calls, factory = patched_requests
    import requests

    monkeypatch.setattr(requests, "head", factory(401))
    report = check_huggingface_access(token="hf_invalid")
    assert all(r.status == "invalid_token" for r in report.models)


def test_unexpected_status_returns_unreachable(patched_requests, monkeypatch):
    _calls, factory = patched_requests
    import requests

    monkeypatch.setattr(requests, "head", factory(503))
    report = check_huggingface_access(token="hf_real")
    assert all(r.status == "unreachable" for r in report.models)


def test_request_exception_returns_unreachable(monkeypatch):
    import requests

    def raising_head(*args, **kwargs):
        raise requests.RequestException("DNS fail")

    monkeypatch.setattr(requests, "head", raising_head)
    report = check_huggingface_access(token="hf_real")
    assert all(r.status == "unreachable" for r in report.models)
    assert all("DNS fail" in r.detail for r in report.models)


def test_log_preflight_emits_warning_for_needs_terms(caplog):
    report = PreflightReport(
        models=[
            ModelAccessResult(model=GATED_MODELS[0], status="needs_terms", detail="x"),
            ModelAccessResult(model=GATED_MODELS[1], status="ok"),
        ]
    )
    with caplog.at_level("WARNING", logger="core.hf_preflight"):
        log_preflight_report(report)
    text = "\n".join(r.message for r in caplog.records)
    assert "gated model" in text.lower() or "accept the terms" in text.lower()
    assert GATED_MODELS[0] in text


def test_log_preflight_emits_error_for_invalid_token(caplog):
    report = PreflightReport(
        models=[
            ModelAccessResult(model=GATED_MODELS[0], status="invalid_token", detail="bad"),
            ModelAccessResult(model=GATED_MODELS[1], status="invalid_token", detail="bad"),
        ]
    )
    with caplog.at_level("WARNING", logger="core.hf_preflight"):
        log_preflight_report(report)
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, "invalid_token must surface at ERROR severity"
    assert any("HF_TOKEN" in r.message for r in error_records)


def test_log_preflight_silent_happy_path_emits_single_info(caplog):
    report = PreflightReport(
        models=[ModelAccessResult(model=m, status="ok") for m in GATED_MODELS]
    )
    with caplog.at_level("INFO", logger="core.hf_preflight"):
        log_preflight_report(report)
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_records) == 1
    assert "all gated models accessible" in info_records[0].message.lower()


def test_uses_env_var_when_token_arg_omitted(monkeypatch, patched_requests):
    _calls, factory = patched_requests
    import requests

    monkeypatch.setattr(requests, "head", factory(200))
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    report = check_huggingface_access()
    assert report.all_ok is True
