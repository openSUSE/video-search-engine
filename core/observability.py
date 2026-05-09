"""Per-request observability primitives.

A single ``ContextVar`` holds the current request id so that any logger
call deep in a request — search service, ChromaDB layer, schema validator —
can attach the id without threading it through every signature. The API
middleware sets the id on entry and clears it on exit; CLI/worker code
paths leave it ``None``, which renders as ``"-"`` in logs.

This module is deliberately self-contained: no FastAPI / Prometheus
imports here. The middleware in ``api.main`` and the formatter in
``core.logger`` consume from this module so unit tests don't need either
dependency to validate the contract.
"""

from __future__ import annotations

import contextvars
import uuid
from typing import Iterator
from contextlib import contextmanager


REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "video_se_request_id", default=None
)


def current_request_id() -> str | None:
    """Return the request id bound to the running task, or ``None``."""
    return _REQUEST_ID.get()


def new_request_id() -> str:
    """Mint a fresh request id (12-char hex prefix of a uuid4)."""
    return uuid.uuid4().hex[:12]


@contextmanager
def request_id_context(request_id: str | None) -> Iterator[str | None]:
    """Bind ``request_id`` for the duration of the ``with`` block.

    Passing ``None`` is a no-op binding (kept for symmetry with callers
    that may not have an id) but still resets the contextvar at exit, so
    nested calls don't leak state.
    """
    token = _REQUEST_ID.set(request_id)
    try:
        yield request_id
    finally:
        _REQUEST_ID.reset(token)
