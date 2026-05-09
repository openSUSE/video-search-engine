"""Opt-in security primitives for the search API.

The repo's threat model assumes the API runs inside a private network
behind ingress, so all of these are off by default — turning them on is a
single env-var change. Three knobs:

* ``SEARCH_API_KEY``: when set, ``/search`` requires the same value in the
  ``X-API-Key`` header. ``/healthz``, ``/readyz``, ``/metrics`` stay open
  so liveness probes and Prometheus scrapers don't need to know the key.
* ``CORS_ALLOW_ORIGINS``: comma-separated origin list. When unset, no CORS
  middleware is attached (browsers can't cross-origin call us at all).
  When set, FastAPI's CORSMiddleware is mounted with that allow-list.
* ``MAX_REQUEST_BODY_BYTES``: hard cap on ``Content-Length`` for state-
  changing endpoints. Default 32 KiB — far above any legitimate ``/search``
  body — so a misbehaving client can't OOM the worker.

Each helper is independently testable and the API wires them together in
``api.main``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Deque, Dict, Iterable, Tuple

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"
DEFAULT_MAX_BODY_BYTES = 32 * 1024  # 32 KiB; ample for /search payloads
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60.0
# When a client exceeds the limit and we don't yet know when their oldest
# request will age out, fall back to retrying after this many seconds.
RATE_LIMIT_FALLBACK_RETRY_AFTER = 5


def _clean_string(value: str | None) -> str:
    return (value or "").strip()


def configured_api_key() -> str:
    """Return the value of ``SEARCH_API_KEY`` or empty string when unset."""
    return _clean_string(os.getenv("SEARCH_API_KEY"))


def configured_cors_origins() -> list[str]:
    """Parse ``CORS_ALLOW_ORIGINS`` into a list of trimmed entries.

    An empty / unset env var means CORS middleware should not be installed
    at all — *not* "allow everywhere". The caller treats an empty list as
    "skip the middleware".
    """
    raw = _clean_string(os.getenv("CORS_ALLOW_ORIGINS"))
    if not raw:
        return []
    return [item for item in (entry.strip() for entry in raw.split(",")) if item]


def configured_rate_limit_per_minute() -> int:
    """Return the per-minute /search request budget per client. 0 = disabled."""
    raw = _clean_string(os.getenv("RATE_LIMIT_PER_MINUTE"))
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid RATE_LIMIT_PER_MINUTE=%r; rate limiting disabled.", raw
        )
        return 0
    if value < 0:
        logger.warning(
            "Negative RATE_LIMIT_PER_MINUTE=%d; rate limiting disabled.", value
        )
        return 0
    return value


def configured_max_body_bytes() -> int:
    raw = _clean_string(os.getenv("MAX_REQUEST_BODY_BYTES"))
    if not raw:
        return DEFAULT_MAX_BODY_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid MAX_REQUEST_BODY_BYTES=%r; using default %d.",
            raw,
            DEFAULT_MAX_BODY_BYTES,
        )
        return DEFAULT_MAX_BODY_BYTES
    if value <= 0:
        logger.warning(
            "MAX_REQUEST_BODY_BYTES=%d is non-positive; using default %d.",
            value,
            DEFAULT_MAX_BODY_BYTES,
        )
        return DEFAULT_MAX_BODY_BYTES
    return value


_OPEN_PATHS = {"/", "/healthz", "/readyz", "/metrics", "/docs", "/openapi.json", "/redoc"}


class _SlidingWindowRateLimiter:
    """In-process per-key sliding-window limiter.

    Each call to ``check`` either records a hit and returns ``(True, 0)``,
    or rejects with ``(False, retry_after_seconds)`` when the caller has
    already used their full budget within the window. Storage is a deque
    of timestamps per key; old entries are popped lazily on each call so
    long-idle keys don't accumulate memory.

    This is deliberately not Redis-backed: a single API replica per pod is
    the common deployment shape here, and an in-process limiter keeps the
    hot path free of network round-trips. For multi-replica deployments,
    swap the implementation for an external store; the public ``check``
    contract stays the same.
    """

    __slots__ = ("_limit", "_window", "_buckets", "_lock")

    def __init__(self, limit: int, window_seconds: float) -> None:
        self._limit = max(0, limit)
        self._window = max(0.001, window_seconds)
        self._buckets: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def window_seconds(self) -> float:
        return self._window

    def check(self, key: str, *, now: float | None = None) -> Tuple[bool, int]:
        if self._limit == 0:
            return True, 0
        timestamp = now if now is not None else time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = deque()
                self._buckets[key] = bucket
            cutoff = timestamp - self._window
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self._limit:
                # Retry after the oldest entry ages out.
                wait = bucket[0] + self._window - timestamp
                return False, max(1, int(wait) + 1)
            bucket.append(timestamp)
            return True, 0


def _client_key(request: Request) -> str:
    """Stable rate-limit key per client.

    Prefer the API key when present (one budget per credential, surviving
    NAT and load balancers). Fall back to the X-Forwarded-For first hop or
    the direct peer IP. Used solely as a bucket key — no security
    decisions hang off it.
    """
    api_key = request.headers.get(API_KEY_HEADER, "").strip()
    if api_key:
        return f"key:{api_key}"
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if forwarded:
        return f"ip:{forwarded}"
    client = request.client
    return f"ip:{client.host}" if client else "ip:unknown"


def _audit_payload(request: Request, *, provided: str, outcome: str) -> dict:
    """Build the structured audit record for one auth attempt.

    Never includes the presented key — only its first 4 chars (or
    ``"-"`` when absent). The full key would be PII / secret material in
    a leak scenario; the prefix is enough to correlate suspicious traffic
    against a known credential rotation timeline.
    """
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    client_host = (request.client.host if request.client else None) or "-"
    return {
        "audit": "search_api_auth",
        "outcome": outcome,
        "method": request.method,
        "path": request.url.path,
        "peer_ip": client_host,
        "forwarded_ip": forwarded or "-",
        "key_presented": bool(provided),
        "key_prefix": (provided[:4] + "…") if provided else "-",
    }


def _install_api_key_auth(app: FastAPI, *, api_key: str) -> None:
    audit_logger = logging.getLogger("api.security.audit")

    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        # Liveness/readiness/metrics need to stay open for probes and scrapes.
        if request.url.path in _OPEN_PATHS or request.method == "OPTIONS":
            return await call_next(request)
        provided = request.headers.get(API_KEY_HEADER, "").strip()
        if provided != api_key:
            # Structured audit line — JSON-mode log shippers can index on
            # outcome=denied to alert on credential brute-forcing. The
            # request id from the observability middleware is already on
            # the record via the contextvar filter.
            audit_logger.warning(
                "auth denied",
                extra=_audit_payload(request, provided=provided, outcome="denied"),
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key."},
                headers={"WWW-Authenticate": API_KEY_HEADER},
            )
        # INFO-level for granted; ops can dial it down to WARNING in prod
        # if the volume becomes noisy, while denials always surface.
        audit_logger.info(
            "auth granted",
            extra=_audit_payload(request, provided=provided, outcome="granted"),
        )
        return await call_next(request)


def _install_rate_limit(app: FastAPI, *, limit: int) -> None:
    limiter = _SlidingWindowRateLimiter(limit, DEFAULT_RATE_LIMIT_WINDOW_SECONDS)
    app.state.rate_limiter = limiter

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if request.url.path in _OPEN_PATHS or request.method == "OPTIONS":
            return await call_next(request)
        allowed, retry_after = limiter.check(_client_key(request))
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded ({limiter.limit} req per "
                        f"{int(limiter.window_seconds)}s). Retry after {retry_after}s."
                    )
                },
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)


def _install_body_size_limit(app: FastAPI, *, limit: int) -> None:
    @app.middleware("http")
    async def body_size_middleware(request: Request, call_next):
        # Reject early on declared Content-Length; for chunked requests
        # without a length, FastAPI's own request body parsing will still
        # bound things via the worker config. We intentionally don't drain
        # the body here: that defeats the point of the early reject.
        content_length_header = request.headers.get("content-length")
        if content_length_header:
            try:
                content_length = int(content_length_header)
            except ValueError:
                content_length = None
            if content_length is not None and content_length > limit:
                # Starlette renamed the constant in newer releases; the
                # numeric value is the same. Hardcode 413 to avoid a
                # DeprecationWarning that triggers on every oversized
                # request.
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body exceeds {limit} bytes."},
                )
        return await call_next(request)


def _install_cors(app: FastAPI, *, origins: Iterable[str]) -> None:
    """Mount CORSMiddleware only when origins is non-empty."""
    from fastapi.middleware.cors import CORSMiddleware

    origins = list(origins)
    if not origins:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", API_KEY_HEADER, "X-Request-ID"],
        allow_credentials=False,
        max_age=600,
    )


def register(app: FastAPI) -> None:
    """Install whichever security middleware the env vars enable.

    Order matters in FastAPI: middleware added later runs *first* on the
    request, so we add CORS last (it should be the outermost layer to
    handle preflight OPTIONS before auth gets a chance to reject them).
    Body-size goes innermost so it short-circuits before auth allocates.
    """
    body_limit = configured_max_body_bytes()
    _install_body_size_limit(app, limit=body_limit)

    # Order: rate-limit before auth so a flood of bad-key requests doesn't
    # also use up the auth check's per-request work — the limiter rejects
    # them earlier in the middleware chain. (FastAPI runs later-added
    # middleware first, which is why the install order here is reversed
    # from the intuitive request flow.)
    api_key = configured_api_key()
    if api_key:
        _install_api_key_auth(app, api_key=api_key)
        logger.info("API key authentication is enabled.")
    else:
        logger.info(
            "SEARCH_API_KEY is not set; /search is reachable without auth. "
            "Set SEARCH_API_KEY in production to require X-API-Key."
        )

    rate_limit = configured_rate_limit_per_minute()
    if rate_limit > 0:
        _install_rate_limit(app, limit=rate_limit)
        logger.info("Rate limiting enabled: %d req/min per client.", rate_limit)

    origins = configured_cors_origins()
    _install_cors(app, origins=origins)
    if origins:
        logger.info("CORS allow_origins = %s", origins)
