"""Reporting helpers for the benchmark suite.

The runner separates *measurement* (the harness) from *presentation* (this
module) so that adding a new output format only requires a new function here.
Two reporters ship out of the box:

* :func:`format_text_report` — a fixed-width table grouped by category.
  Tuned for terminals at least 110 columns wide.
* :func:`format_json_report` — a deterministic JSON document suitable for
  diffing across runs and feeding into long-running trend dashboards.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Mapping

from .harness import BenchmarkResult


@dataclass(frozen=True)
class RunMetadata:
    """Lightweight context attached to JSON reports.

    Capturing the interpreter, platform, and clock id makes it possible to
    spot regressions caused by environment changes rather than code changes.
    """

    python_version: str
    platform: str
    timestamp_unix: float
    scale: float

    @classmethod
    def capture(cls, scale: float) -> "RunMetadata":
        return cls(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            timestamp_unix=time.time(),
            scale=scale,
        )

    def to_dict(self) -> Mapping[str, object]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "timestamp_unix": self.timestamp_unix,
            "scale": self.scale,
        }


def _format_ns(value: float) -> str:
    """Render a nanosecond value in the most readable unit."""

    if value != value:  # NaN
        return "nan"
    if value < 1_000:
        return f"{value:7.1f}ns"
    if value < 1_000_000:
        return f"{value / 1_000:7.2f}µs"
    if value < 1_000_000_000:
        return f"{value / 1_000_000:7.2f}ms"
    return f"{value / 1_000_000_000:7.2f}s"


def _format_ops(value: float) -> str:
    if value == float("inf"):
        return "      inf"
    if value >= 1_000_000:
        return f"{value / 1_000_000:7.2f}M/s"
    if value >= 1_000:
        return f"{value / 1_000:7.2f}K/s"
    return f"{value:7.1f}/s"


def format_text_report(
    results: Iterable[BenchmarkResult], *, metadata: RunMetadata
) -> str:
    """Return a multi-line text table summarising ``results``."""

    grouped: dict[str, List[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.category, []).append(result)

    lines: List[str] = []
    lines.append("Video Search Engine — Benchmark Report")
    lines.append(
        f"  python={metadata.python_version} platform={metadata.platform} "
        f"scale={metadata.scale:g}"
    )
    lines.append("")

    header = (
        f"{'name':40s}  {'iters':>7s}  {'min':>9s}  {'median':>9s}  "
        f"{'mean':>9s}  {'p95':>9s}  {'p99':>9s}  {'stdev':>9s}  {'ops':>10s}"
    )

    for category in sorted(grouped):
        lines.append(f"[{category}]")
        lines.append(header)
        lines.append("-" * len(header))
        for result in sorted(grouped[category], key=lambda r: r.name):
            lines.append(
                f"{result.name[:40]:40s}  {result.iterations:7d}  "
                f"{_format_ns(result.min_ns):>9s}  "
                f"{_format_ns(result.median_ns):>9s}  "
                f"{_format_ns(result.mean_ns):>9s}  "
                f"{_format_ns(result.p95_ns):>9s}  "
                f"{_format_ns(result.p99_ns):>9s}  "
                f"{_format_ns(result.stdev_ns):>9s}  "
                f"{_format_ops(result.ops_per_second):>10s}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def format_json_report(
    results: Iterable[BenchmarkResult], *, metadata: RunMetadata
) -> str:
    """Return a JSON string with deterministic key order for diffing."""

    payload = {
        "metadata": metadata.to_dict(),
        "results": [result.to_dict() for result in results],
    }
    return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"


def format_markdown_report(
    results: Iterable[BenchmarkResult], *, metadata: RunMetadata
) -> str:
    """Return a GitHub-flavoured markdown report.

    Suitable for pasting into PR comments or README snippets — categories
    become H3 headers and benchmarks become rows in a markdown table. Units
    are picked per-cell so a table mixing nanoseconds and milliseconds
    remains readable.
    """

    grouped: dict[str, List[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.category, []).append(result)

    lines: List[str] = []
    lines.append("# Video Search Engine — Benchmark Report")
    lines.append("")
    lines.append(
        f"- python: `{metadata.python_version}`"
    )
    lines.append(f"- platform: `{metadata.platform}`")
    lines.append(f"- scale: `{metadata.scale:g}`")
    lines.append("")

    for category in sorted(grouped):
        lines.append(f"### {category}")
        lines.append("")
        lines.append(
            "| benchmark | iters | min | median | mean | p95 | p99 | stdev | ops/s |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for result in sorted(grouped[category], key=lambda r: r.name):
            lines.append(
                "| `{name}` | {iters} | {min} | {med} | {mean} | "
                "{p95} | {p99} | {stdev} | {ops} |".format(
                    name=result.name,
                    iters=result.iterations,
                    min=_format_ns(result.min_ns).strip(),
                    med=_format_ns(result.median_ns).strip(),
                    mean=_format_ns(result.mean_ns).strip(),
                    p95=_format_ns(result.p95_ns).strip(),
                    p99=_format_ns(result.p99_ns).strip(),
                    stdev=_format_ns(result.stdev_ns).strip(),
                    ops=_format_ops(result.ops_per_second).strip(),
                )
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
