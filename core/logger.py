import json
import logging
import os
from typing import Any

import colorlog

from core.observability import current_request_id

_VALID_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
_VALID_FORMATS = {"pretty", "json"}
_PROJECT_LOG_HANDLER_ATTR = "_video_search_engine_handler"


def _resolve_log_level() -> int:
    """Return the log level requested via the LOG_LEVEL env var, or INFO."""

    raw = (os.getenv("LOG_LEVEL") or "").strip().upper()
    if raw in _VALID_LEVELS:
        return getattr(logging, raw)
    return logging.INFO


def _resolve_log_format() -> str:
    """Return ``json`` or ``pretty``. Default ``pretty`` for human local dev.

    Production deployments should set ``LOG_FORMAT=json`` so each log line
    is a parseable record (request_id, level, logger, message).
    """
    raw = (os.getenv("LOG_FORMAT") or "").strip().lower()
    if raw in _VALID_FORMATS:
        return raw
    return "pretty"


class _RequestIdFilter(logging.Filter):
    """Attach the contextvar request id (if any) onto every record.

    Records that already carry a ``request_id`` attribute (e.g. set
    explicitly by a caller) are left alone.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id() or "-"
        return True


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter — one record per line, deterministic key order."""

    _RESERVED_FIELDS = {
        "args",
        "msg",
        "name",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "asctime",
        "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        # Build the message safely; %-formatting can fail on unusual args.
        try:
            message = record.getMessage()
        except Exception as exc:  # pragma: no cover - defensive
            message = f"<log format error: {exc!r} raw={record.msg!r}>"

        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
            "request_id": getattr(record, "request_id", "-"),
        }
        # Surface any caller-supplied extra fields (``logger.info(..., extra={"k":"v"})``)
        # so structured-log consumers can use them. Reserved attributes set
        # by Python's logging itself are excluded.
        for key, value in record.__dict__.items():
            if key in self._RESERVED_FIELDS or key in payload:
                continue
            if key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _build_formatter(log_format: str) -> logging.Formatter:
    if log_format == "json":
        return _JsonFormatter()
    # Pretty (colored) format for local development; the request id slot
    # renders ``-`` when no request is in flight.
    return colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - [%(name)s] req=%(request_id)s - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )


def setup_logging():
    """
    Sets up centralized logging with optional JSON output and a per-record
    request-id contextvar. Idempotent: callable at every entrypoint without
    duplicating handlers.
    """
    root_logger = logging.getLogger()

    # Avoid adding handlers multiple times if this function is called again.
    if any(
        getattr(handler, _PROJECT_LOG_HANDLER_ATTR, False)
        for handler in root_logger.handlers
    ):
        return

    root_logger.setLevel(_resolve_log_level())

    if root_logger.hasHandlers():
        return

    log_format = _resolve_log_format()
    formatter = _build_formatter(log_format)

    if log_format == "json":
        console_handler: logging.Handler = logging.StreamHandler()
    else:
        console_handler = colorlog.StreamHandler()
    setattr(console_handler, _PROJECT_LOG_HANDLER_ATTR, True)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_RequestIdFilter())

    root_logger.addHandler(console_handler)

    log = logging.getLogger(__name__)
    log.info("Logging configured (format=%s).", log_format)
