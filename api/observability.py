"""FastAPI-side observability wiring.

Splits two concerns out of ``api.main`` so the latter stays a thin route
file:

* a request-id middleware that pulls or mints ``X-Request-ID`` and binds
  it via ``core.observability``;
* prometheus_client metrics — counters and a histogram for /search,
  exposed at ``/metrics``.

The middleware is order-sensitive: it must wrap the handler so the
contextvar binding is live for the duration of the request, including
exception logs. The metrics module is imported lazily inside ``register``
so unit tests that don't need it (and don't want a global registry pollution)
can skip the wiring.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, Request, Response

from core.observability import (
    REQUEST_ID_HEADER,
    new_request_id,
    request_id_context,
)


logger = logging.getLogger(__name__)


def _normalize_route(path: str) -> str:
    """Collapse ids and query strings out of a path for low-cardinality labels.

    Without this, every distinct request path becomes a unique label
    combination and the prometheus_client cardinality blows up. We keep
    the registered route templates (e.g. ``/search``) and bucket
    everything else into ``/_other`` rather than emitting raw paths.
    """
    if not path:
        return "/_other"
    head = path.split("?", 1)[0]
    if head in {"/", "/healthz", "/readyz", "/search", "/metrics", "/docs", "/openapi.json"}:
        return head
    return "/_other"


def _install_request_id_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Any) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "").strip()
        request_id = incoming or new_request_id()
        with request_id_context(request_id):
            try:
                response = await call_next(request)
            except Exception:
                # Re-raise so FastAPI's exception handlers run; we just
                # want the contextvar bound when the error logs fire.
                logger.exception(
                    "Unhandled exception while processing %s %s",
                    request.method,
                    request.url.path,
                )
                raise
            response.headers[REQUEST_ID_HEADER] = request_id
            return response


def _install_metrics(app: FastAPI) -> None:
    try:
        from prometheus_client import (
            CONTENT_TYPE_LATEST,
            CollectorRegistry,
            Counter,
            Histogram,
            generate_latest,
        )
    except ImportError:
        logger.warning(
            "prometheus_client not installed; /metrics endpoint disabled."
        )
        return

    # Use a per-app registry so test instances don't share global state.
    registry = CollectorRegistry()
    request_counter = Counter(
        "video_se_http_requests_total",
        "HTTP requests by route, method, and status class.",
        labelnames=("route", "method", "status"),
        registry=registry,
    )
    request_latency = Histogram(
        "video_se_http_request_duration_seconds",
        "HTTP request duration in seconds.",
        labelnames=("route", "method"),
        registry=registry,
    )
    search_outcomes = Counter(
        "video_se_search_requests_total",
        "Search requests by outcome (success|error).",
        labelnames=("outcome",),
        registry=registry,
    )
    search_results = Histogram(
        "video_se_search_results_count",
        "Distribution of result counts returned by /search.",
        buckets=(0, 1, 3, 5, 10, 20, 50),
        registry=registry,
    )

    app.state.metrics = {
        "request_counter": request_counter,
        "request_latency": request_latency,
        "search_outcomes": search_outcomes,
        "search_results": search_results,
    }

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Any) -> Response:
        route = _normalize_route(request.url.path)
        # Skip /metrics itself so a dashboard scraping us doesn't dominate
        # its own counters.
        if route == "/metrics":
            return await call_next(request)
        method = request.method.upper()
        start = time.monotonic()
        status_class = "5xx"
        try:
            response = await call_next(request)
        except Exception:
            request_counter.labels(route=route, method=method, status=status_class).inc()
            request_latency.labels(route=route, method=method).observe(
                time.monotonic() - start
            )
            raise
        status_class = f"{response.status_code // 100}xx"
        request_counter.labels(route=route, method=method, status=status_class).inc()
        request_latency.labels(route=route, method=method).observe(
            time.monotonic() - start
        )
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(
            content=generate_latest(registry),
            media_type=CONTENT_TYPE_LATEST,
        )


def register(app: FastAPI) -> None:
    """Install the request-id middleware and (when available) /metrics."""
    _install_request_id_middleware(app)
    _install_metrics(app)
