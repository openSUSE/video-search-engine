"""Benchmarks for ``core.logger`` and Python logging hot paths.

``setup_logging`` runs once per process, but every CLI entry point and worker
calls it. The function early-returns when handlers are already attached, so
the cost we want to track is the *first* attachment (slow path) and the
re-entry case (must stay near zero). We also benchmark a simple ``logger.info``
call because the ingestion pipeline emits dozens of them per segment.
"""

from __future__ import annotations

import contextlib
import io
import logging
from typing import Any

from core import logger as core_logger

from .harness import Benchmark


def _setup_first_attachment():
    # The harness benchmarks the slow path of setup_logging: handlers cleared
    # before each call so setup_logging re-attaches a ColoredFormatter and
    # StreamHandler. The "Colored logging is configured." info line emitted
    # by setup_logging is captured and discarded so the report stays clean.
    return io.StringIO()


def _bench_setup_logging_first_call(buffer: io.StringIO) -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    with contextlib.redirect_stderr(buffer):
        core_logger.setup_logging()
    buffer.truncate(0)
    buffer.seek(0)


def _bench_setup_logging_idempotent(_: Any) -> None:
    # Hot path: handlers already attached, function returns immediately.
    core_logger.setup_logging()


def _setup_logger_info():
    logger = logging.getLogger("benchmark.bench_logger")
    # Detach the colorlog handler so the benchmark does not pay its cost on
    # every iteration — we want to measure the dispatch cost, not the
    # formatter cost.
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def _bench_logger_info(logger: logging.Logger) -> None:
    logger.info("processed segment %s in %sms", "segment_0001", 12)


BENCHMARKS = [
    Benchmark(
        name="logger.setup_logging(first call)",
        category="config",
        description=(
            "First-call setup_logging that allocates a colorlog handler. "
            "The benchmark resets handlers each iteration."
        ),
        fn=_bench_setup_logging_first_call,
        setup=_setup_first_attachment,
        iterations=400,
    ),
    Benchmark(
        name="logger.setup_logging(idempotent)",
        category="config",
        description="Re-entrant setup_logging() returns early when configured.",
        fn=_bench_setup_logging_idempotent,
        iterations=50_000,
    ),
    Benchmark(
        name="logging.Logger.info(NullHandler)",
        category="config",
        description=(
            "Baseline logger.info dispatch with a NullHandler — measures the "
            "cost the ingestion pipeline pays per emitted log line."
        ),
        fn=_bench_logger_info,
        setup=_setup_logger_info,
        iterations=50_000,
    ),
]
