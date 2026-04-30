"""Tests for ``benchmarks.compare`` and the markdown reporter.

The comparison logic is what makes the benchmark suite useful for trend
tracking and regression gating — the unit tests below pin its behaviour at
the boundary cases (added/removed/threshold/zero baseline).
"""

from __future__ import annotations

import json

import pytest

from benchmarks.compare import (
    compare_reports,
    format_comparison,
    has_regression,
    load_report,
)
from benchmarks.harness import BenchmarkResult
from benchmarks.reporters import RunMetadata, format_markdown_report


def _report(*entries):
    return {
        "metadata": {
            "python_version": "3.12",
            "platform": "test",
            "timestamp_unix": 0.0,
            "scale": 1.0,
        },
        "results": [
            {
                "name": name,
                "category": category,
                "description": "",
                "iterations": 10,
                "inner_loops": 1,
                "ops_per_second": 0.0,
                "ns": {
                    "min": median,
                    "median": median,
                    "mean": median,
                    "p95": median,
                    "p99": median,
                    "max": median,
                    "stdev": 0.0,
                    "total": median * 10,
                },
            }
            for (name, category, median) in entries
        ],
    }


def test_compare_reports_flags_regression_past_threshold():
    baseline = _report(("a", "api", 100.0))
    current = _report(("a", "api", 130.0))  # +30% > 10% default warn

    rows = compare_reports(baseline, current, warn_ratio=0.10)

    assert len(rows) == 1
    row = rows[0]
    assert row.status == "regression"
    assert row.delta_ratio == pytest.approx(0.30)
    assert has_regression(rows) is True


def test_compare_reports_flags_improvement_past_threshold():
    baseline = _report(("a", "api", 100.0))
    current = _report(("a", "api", 70.0))  # -30%

    rows = compare_reports(baseline, current, warn_ratio=0.10)

    assert rows[0].status == "improvement"
    assert rows[0].delta_ratio == pytest.approx(-0.30)
    assert has_regression(rows) is False


def test_compare_reports_unchanged_within_threshold():
    baseline = _report(("a", "api", 100.0))
    current = _report(("a", "api", 105.0))  # +5%, below warn

    rows = compare_reports(baseline, current, warn_ratio=0.10)

    assert rows[0].status == "unchanged"


def test_compare_reports_added_and_removed():
    baseline = _report(("kept", "api", 100.0), ("only_baseline", "api", 50.0))
    current = _report(("kept", "api", 102.0), ("only_current", "api", 80.0))

    rows = compare_reports(baseline, current, warn_ratio=0.10)
    by_name = {row.name: row for row in rows}

    assert by_name["kept"].status == "unchanged"
    assert by_name["only_baseline"].status == "removed"
    assert by_name["only_baseline"].current_median_ns is None
    assert by_name["only_current"].status == "added"
    assert by_name["only_current"].baseline_median_ns is None


def test_compare_reports_zero_baseline_is_unchanged():
    baseline = _report(("a", "api", 0.0))
    current = _report(("a", "api", 100.0))

    rows = compare_reports(baseline, current, warn_ratio=0.10)

    assert rows[0].status == "unchanged"
    assert rows[0].delta_ratio is None


@pytest.mark.parametrize("median", [float("nan"), float("inf"), -1.0])
def test_compare_reports_ignores_unusable_baseline_medians(median):
    baseline = _report(("a", "api", median))
    current = _report(("a", "api", 100.0))

    rows = compare_reports(baseline, current, warn_ratio=0.10)

    assert rows[0].status == "unchanged"
    assert rows[0].baseline_median_ns is None
    assert rows[0].delta_ratio is None


@pytest.mark.parametrize("median", [float("nan"), float("inf"), -1.0])
def test_compare_reports_ignores_unusable_current_medians(median):
    baseline = _report(("a", "api", 100.0))
    current = _report(("a", "api", median))

    rows = compare_reports(baseline, current, warn_ratio=0.10)

    assert rows[0].status == "unchanged"
    assert rows[0].current_median_ns is None
    assert rows[0].delta_ratio is None


@pytest.mark.parametrize("warn_ratio", [-0.5, float("nan"), float("inf")])
def test_compare_reports_rejects_unusable_warn_ratio(warn_ratio):
    baseline = _report(("a", "api", 100.0))
    current = _report(("a", "api", 100.0))

    with pytest.raises(ValueError, match="warn_ratio"):
        compare_reports(baseline, current, warn_ratio=warn_ratio)


def test_compare_reports_rejects_invalid_payload():
    with pytest.raises(ValueError):
        compare_reports({"results": "not-a-list"}, _report(), warn_ratio=0.1)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ("not-a-result", "result at index 0"),
        ({"category": "api", "ns": {"median": 100.0}}, "name"),
        ({"name": "", "category": "api", "ns": {"median": 100.0}}, "name"),
        ({"name": "a", "category": 7, "ns": {"median": 100.0}}, "category"),
        ({"name": "a", "category": "api", "ns": []}, "ns"),
    ],
)
def test_compare_reports_rejects_malformed_result_entries(entry, message):
    with pytest.raises(ValueError, match=message):
        compare_reports({"results": [entry]}, _report(), warn_ratio=0.1)


def test_compare_reports_rejects_duplicate_result_names():
    report = _report(("a", "api", 100.0), ("a", "api", 110.0))

    with pytest.raises(ValueError, match="duplicate benchmark result name"):
        compare_reports(report, _report(), warn_ratio=0.1)


def test_format_comparison_summary_counts_regressions():
    baseline = _report(
        ("regressed", "api", 100.0),
        ("improved", "api", 100.0),
        ("flat", "api", 100.0),
    )
    current = _report(
        ("regressed", "api", 200.0),
        ("improved", "api", 50.0),
        ("flat", "api", 101.0),
    )

    rows = compare_reports(baseline, current, warn_ratio=0.10)
    text = format_comparison(rows, warn_ratio=0.10)

    assert "1 regression(s)" in text
    assert "1 improvement(s)" in text
    assert "regression" in text
    assert "improvement" in text


def test_load_report_round_trips(tmp_path):
    payload = _report(("a", "api", 100.0))
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(payload))

    loaded = load_report(report_path)

    assert loaded == payload


def test_format_markdown_report_emits_table():
    metadata = RunMetadata.capture(scale=1.0)
    result = BenchmarkResult(
        name="example.bench",
        category="api",
        description="",
        iterations=5,
        inner_loops=1,
        samples_ns=[100, 110, 120, 130, 140],
    )

    text = format_markdown_report([result], metadata=metadata)

    assert text.startswith("# Video Search Engine")
    assert "### api" in text
    assert "| `example.bench` |" in text
    # Columns header has the right number of separators
    assert "| --- | ---: |" in text
