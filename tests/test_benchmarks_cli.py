"""End-to-end CLI tests for ``python -m benchmarks.runner``.

The unit tests in ``test_benchmarks_harness`` cover the discovery, filter,
and reporter logic by calling :func:`benchmarks.runner.main` in-process. The
subprocess tests in this module exercise the parts that only happen in a
real invocation:

* ``sys.argv`` parsing and ``argparse`` exit codes.
* The ``stdout`` / ``stderr`` split (text report on stdout, progress on
  stderr).
* The ``--json`` file-write side-effect.
* The ``--baseline`` + ``--fail-on-regression`` exit code.

A small ``--scale`` keeps each subprocess under one second on a developer
laptop. The suite is otherwise the same code path the user runs locally
and that ``make bench-smoke`` runs in CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_runner(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "benchmarks.runner", *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_runner_list_exits_zero_and_prints_categories():
    result = _run_runner("--list")

    assert result.returncode == 0, result.stderr
    assert "[api]" in result.stdout
    assert "[ingestion]" in result.stdout
    assert "[ui]" in result.stdout


def test_runner_filter_with_json_writes_file(tmp_path):
    out = tmp_path / "report.json"
    result = _run_runner(
        "--scale",
        "0.02",
        "--quiet",
        "--filter",
        "^jobs\\.",
        "--json",
        str(out),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["metadata"]["scale"] == 0.02
    names = [entry["name"] for entry in payload["results"]]
    assert names, "expected at least one job benchmark in the report"
    assert all(name.startswith("jobs.") for name in names)


def test_runner_json_creates_parent_directories(tmp_path):
    out = tmp_path / "nested" / "reports" / "report.json"
    result = _run_runner(
        "--scale",
        "0.02",
        "--quiet",
        "--filter",
        "^jobs\\.encode_job_message\\(minimal\\)$",
        "--json",
        str(out),
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["results"][0]["name"] == (
        "jobs.encode_job_message(minimal)"
    )


def test_runner_json_write_failure_reports_usage_error(tmp_path):
    result = _run_runner(
        "--scale",
        "0.02",
        "--quiet",
        "--filter",
        "^jobs\\.encode_job_message\\(minimal\\)$",
        "--json",
        str(tmp_path),
    )

    assert result.returncode == 2
    assert "Could not write benchmark JSON report" in result.stderr
    assert "Traceback" not in result.stderr


def test_runner_format_json_writes_to_stdout(tmp_path):
    result = _run_runner(
        "--scale",
        "0.02",
        "--quiet",
        "--filter",
        "^jobs\\.encode_job_message\\(minimal\\)$",
        "--format",
        "json",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "metadata" in payload
    assert "results" in payload


def test_runner_format_md_emits_markdown_table():
    result = _run_runner(
        "--scale",
        "0.02",
        "--quiet",
        "--filter",
        "^jobs\\.encode_job_message\\(minimal\\)$",
        "--format",
        "md",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("# Video Search Engine")
    assert "| --- | ---: |" in result.stdout


def test_runner_baseline_with_no_regression_exits_zero(tmp_path):
    baseline = tmp_path / "baseline.json"
    first = _run_runner(
        "--scale",
        "0.02",
        "--quiet",
        "--filter",
        "^jobs\\.",
        "--json",
        str(baseline),
    )
    assert first.returncode == 0, first.stderr

    rerun = _run_runner(
        "--scale",
        "0.02",
        "--quiet",
        "--filter",
        "^jobs\\.",
        "--baseline",
        str(baseline),
        "--warn-ratio",
        "5.0",  # Allow up to 500% drift so noise never trips the test.
        "--fail-on-regression",
    )

    assert rerun.returncode == 0, rerun.stderr
    assert "summary:" in rerun.stderr


def test_runner_baseline_with_forced_regression_exits_one(tmp_path):
    """A baseline of effectively-zero medians forces every benchmark to look slower."""

    forced_baseline = {
        "metadata": {
            "python_version": "synthetic",
            "platform": "synthetic",
            "scale": 1.0,
            "timestamp_unix": 0.0,
        },
        "results": [
            {
                "name": "jobs.encode_job_message(minimal)",
                "category": "jobs",
                "description": "",
                "iterations": 1,
                "inner_loops": 1,
                "ops_per_second": 0.0,
                "ns": {
                    "min": 1.0,
                    "median": 1.0,
                    "mean": 1.0,
                    "p95": 1.0,
                    "p99": 1.0,
                    "max": 1.0,
                    "stdev": 0.0,
                    "total": 1,
                },
            }
        ],
    }

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(forced_baseline))

    result = _run_runner(
        "--scale",
        "0.02",
        "--quiet",
        "--filter",
        "^jobs\\.encode_job_message\\(minimal\\)$",
        "--baseline",
        str(baseline_path),
        "--warn-ratio",
        "0.10",
        "--fail-on-regression",
    )

    assert result.returncode == 1, result.stderr
    assert "regression" in result.stderr


@pytest.mark.parametrize(
    ("baseline_payload", "message"),
    [
        ("{not-json", "valid JSON"),
        (json.dumps([]), "JSON object"),
        (json.dumps({"results": "not-a-list"}), "results"),
        (
            json.dumps(
                {
                    "results": [
                        {
                            "category": "jobs",
                            "ns": {"median": 1.0},
                        }
                    ]
                }
            ),
            "name",
        ),
        (
            json.dumps(
                {
                    "results": [
                        {
                            "name": "jobs.encode_job_message(minimal)",
                            "category": "jobs",
                            "ns": {"median": 1.0},
                        },
                        {
                            "name": "jobs.encode_job_message(minimal)",
                            "category": "jobs",
                            "ns": {"median": 1.0},
                        },
                    ]
                }
            ),
            "duplicate benchmark result name",
        ),
        (
            json.dumps(
                {
                    "results": [
                        {
                            "name": "jobs.encode_job_message(minimal)",
                            "category": "jobs",
                            "ns": {"median": None},
                        }
                    ]
                }
            ),
            "median",
        ),
    ],
)
def test_runner_invalid_baseline_reports_usage_error(
    tmp_path,
    baseline_payload: str,
    message: str,
):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(baseline_payload)

    result = _run_runner(
        "--scale",
        "0.02",
        "--quiet",
        "--filter",
        "^jobs\\.encode_job_message\\(minimal\\)$",
        "--baseline",
        str(baseline),
    )

    assert result.returncode == 2
    assert "Invalid benchmark baseline" in result.stderr
    assert message in result.stderr
    assert "Traceback" not in result.stderr


def test_runner_no_match_filter_exits_two():
    result = _run_runner(
        "--scale",
        "0.02",
        "--filter",
        "this-pattern-matches-nothing",
    )

    assert result.returncode == 2
    assert "No benchmarks selected" in result.stderr


def test_runner_invalid_filter_regex_is_handled():
    result = _run_runner("--filter", "[")

    assert result.returncode == 2
    assert "invalid --filter pattern" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("flag", ["--scale=0", "--scale=-1", "--scale=nan"])
def test_runner_invalid_scale_is_handled(flag: str):
    result = _run_runner(flag, "--quiet", "--filter", "^jobs\\.")

    assert result.returncode == 2
    assert "finite positive number" in result.stderr


@pytest.mark.parametrize("flag", ["--warn-ratio=-0.1", "--warn-ratio=nan", "--warn-ratio=inf"])
def test_runner_invalid_warn_ratio_is_handled(flag: str):
    result = _run_runner(flag, "--list")

    assert result.returncode == 2
    assert "finite non-negative number" in result.stderr
