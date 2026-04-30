import logging

import pytest

from core import logger as core_logger


@pytest.fixture(autouse=True)
def _reset_root_handlers():
    root_logger = logging.getLogger()
    saved_handlers = list(root_logger.handlers)
    saved_level = root_logger.level
    yield
    root_logger.handlers = saved_handlers
    root_logger.setLevel(saved_level)


def _clear_handlers():
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)


def test_setup_logging_defaults_to_info_when_log_level_unset(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    _clear_handlers()

    core_logger.setup_logging()

    assert logging.getLogger().level == logging.INFO


@pytest.mark.parametrize(
    ("env_value", "expected_level"),
    [
        ("DEBUG", logging.DEBUG),
        ("debug", logging.DEBUG),
        ("  WARNING  ", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_setup_logging_honors_log_level_env(monkeypatch, env_value, expected_level):
    monkeypatch.setenv("LOG_LEVEL", env_value)
    _clear_handlers()

    core_logger.setup_logging()

    assert logging.getLogger().level == expected_level


@pytest.mark.parametrize("env_value", ["", "  ", "verbose", "TRACE", "1"])
def test_setup_logging_falls_back_to_info_for_unknown_level(monkeypatch, env_value):
    monkeypatch.setenv("LOG_LEVEL", env_value)
    _clear_handlers()

    core_logger.setup_logging()

    assert logging.getLogger().level == logging.INFO


def test_setup_logging_is_idempotent(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    _clear_handlers()

    core_logger.setup_logging()
    handlers_after_first = list(logging.getLogger().handlers)

    # Even if LOG_LEVEL changes, repeated calls don't reattach.
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    core_logger.setup_logging()

    assert logging.getLogger().handlers == handlers_after_first
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_honors_log_level_with_existing_handler(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    _clear_handlers()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    external_handler = logging.StreamHandler()
    root_logger.addHandler(external_handler)

    core_logger.setup_logging()

    assert root_logger.handlers == [external_handler]
    assert root_logger.level == logging.DEBUG
