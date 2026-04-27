"""Compare two benchmark runs and flag regressions.

The runner can persist a JSON snapshot via ``--json``. This module loads two
such snapshots and produces a structured diff so reviewers can answer one
question quickly: *did this change make anything slower than the agreed
threshold?*

The comparison is intentionally tolerant of churn:

* Benchmarks present only in the baseline are reported as ``"removed"``.
* Benchmarks present only in the current run are reported as ``"added"``.
* Benchmarks present in both compute a relative delta against the baseline
  median (median is more stable than mean for short runs) and a verdict:
  ``"regression"`` past the warn threshold, ``"improvement"`` past the warn
  threshold in the other direction, or ``"unchanged"`` otherwise.

Median is the comparison axis because it ignores the tail, which is dominated
by environmental noise (other processes, GC, allocator) rather than code.
For tail-sensitive analysis, drop into the JSON directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class ComparisonRow:
    """A single row of the comparison report."""

    name: str
    category: str
    status: str  # one of: regression, improvement, unchanged, added, removed
    baseline_median_ns: float | None
    current_median_ns: float | None
    delta_ratio: float | None  # (current - baseline) / baseline; None for added/removed

    def is_regression(self) -> bool:
        return self.status == "regression"


def load_report(path: str | Path) -> Mapping[str, object]:
    """Load a JSON report produced by ``benchmarks.runner --json``."""

    return json.loads(Path(path).read_text())


def _median_by_name(report: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("report must contain a 'results' list")
    indexed: Dict[str, Dict[str, object]] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        ns = entry.get("ns")
        if not isinstance(name, str) or not isinstance(ns, dict):
            continue
        indexed[name] = {
            "category": entry.get("category", "misc"),
            "median": ns.get("median"),
        }
    return indexed


def compare_reports(
    baseline: Mapping[str, object],
    current: Mapping[str, object],
    *,
    warn_ratio: float = 0.10,
) -> List[ComparisonRow]:
    """Diff two reports and return one :class:`ComparisonRow` per benchmark.

    ``warn_ratio`` is the symmetric threshold that turns a delta into a
    regression or improvement. The default ``0.10`` flags any change of more
    than 10% relative to the baseline median.
    """

    if warn_ratio < 0:
        raise ValueError("warn_ratio must be non-negative")

    baseline_index = _median_by_name(baseline)
    current_index = _median_by_name(current)

    rows: List[ComparisonRow] = []
    for name in sorted(set(baseline_index) | set(current_index)):
        in_baseline = name in baseline_index
        in_current = name in current_index

        if in_baseline and not in_current:
            entry = baseline_index[name]
            rows.append(
                ComparisonRow(
                    name=name,
                    category=str(entry["category"]),
                    status="removed",
                    baseline_median_ns=_as_float(entry["median"]),
                    current_median_ns=None,
                    delta_ratio=None,
                )
            )
            continue

        if in_current and not in_baseline:
            entry = current_index[name]
            rows.append(
                ComparisonRow(
                    name=name,
                    category=str(entry["category"]),
                    status="added",
                    baseline_median_ns=None,
                    current_median_ns=_as_float(entry["median"]),
                    delta_ratio=None,
                )
            )
            continue

        baseline_entry = baseline_index[name]
        current_entry = current_index[name]
        baseline_median = _as_float(baseline_entry["median"])
        current_median = _as_float(current_entry["median"])

        if (
            baseline_median is None
            or current_median is None
            or baseline_median <= 0
        ):
            status = "unchanged"
            delta = None
        else:
            delta = (current_median - baseline_median) / baseline_median
            if delta > warn_ratio:
                status = "regression"
            elif delta < -warn_ratio:
                status = "improvement"
            else:
                status = "unchanged"

        rows.append(
            ComparisonRow(
                name=name,
                # Prefer the current category when both exist (categories can
                # be renamed; the latest run is the source of truth).
                category=str(current_entry["category"]),
                status=status,
                baseline_median_ns=baseline_median,
                current_median_ns=current_median,
                delta_ratio=delta,
            )
        )

    return rows


def has_regression(rows: Iterable[ComparisonRow]) -> bool:
    """Return True when any row has a ``"regression"`` status."""

    return any(row.is_regression() for row in rows)


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def format_comparison(
    rows: List[ComparisonRow], *, warn_ratio: float
) -> str:
    """Render a comparison as a fixed-width text table."""

    lines: List[str] = []
    lines.append(
        f"Comparison (warn threshold = {warn_ratio:.0%} change in median)"
    )
    header = (
        f"{'name':40s}  {'category':12s}  {'baseline':>11s}  "
        f"{'current':>11s}  {'delta':>9s}  status"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for row in rows:
        baseline = _format_optional_ns(row.baseline_median_ns)
        current = _format_optional_ns(row.current_median_ns)
        if row.delta_ratio is None:
            delta = "    n/a"
        else:
            delta = f"{row.delta_ratio * 100:+8.2f}%"
        lines.append(
            f"{row.name[:40]:40s}  {row.category[:12]:12s}  "
            f"{baseline:>11s}  {current:>11s}  {delta:>9s}  {row.status}"
        )

    regressions = sum(1 for row in rows if row.is_regression())
    improvements = sum(1 for row in rows if row.status == "improvement")
    lines.append("")
    lines.append(
        f"summary: {regressions} regression(s), {improvements} improvement(s), "
        f"{len(rows)} benchmark(s) compared"
    )
    return "\n".join(lines) + "\n"


def _format_optional_ns(value: float | None) -> str:
    if value is None:
        return "       n/a"
    if value < 1_000:
        return f"{value:7.1f}ns"
    if value < 1_000_000:
        return f"{value / 1_000:7.2f}µs"
    if value < 1_000_000_000:
        return f"{value / 1_000_000:7.2f}ms"
    return f"{value / 1_000_000_000:7.2f}s"
