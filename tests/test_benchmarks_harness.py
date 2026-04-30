"""Tests for the lightweight benchmark harness.

We do not run the actual benchmark suite from CI (timings are not assertable),
but we do guard the harness itself: discovery, statistics, scaling, the
report formatters, and the CLI entry-point. This keeps the suite usable
without paying its full runtime cost on every push.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from benchmarks import runner as runner_module
from benchmarks.harness import Benchmark, BenchmarkResult, run_benchmark
from benchmarks.reporters import RunMetadata, format_json_report, format_text_report


def _noop(_: Any) -> None:
    return None


def _record_calls(payload: list) -> Benchmark:
    def setup() -> list:
        return payload

    def fn(state: list) -> None:
        state.append(1)

    return Benchmark(name="record", fn=fn, setup=setup, iterations=5, warmup=1)


def test_run_benchmark_runs_warmup_then_iterations():
    payload: list = []
    benchmark = _record_calls(payload)

    result = run_benchmark(benchmark)

    assert result.iterations == 5
    # 1 warmup + 5 timed iterations = 6 calls
    assert len(payload) == 6
    assert len(result.samples_ns) == 5
    assert result.name == "record"


def test_run_benchmark_scale_multiplies_iterations():
    payload: list = []
    benchmark = _record_calls(payload)

    result = run_benchmark(benchmark, scale=2.0)
    # 1 warmup + 10 timed iterations
    assert len(payload) == 11
    assert result.iterations == 10


def test_run_benchmark_scale_floor_is_one():
    payload: list = []
    benchmark = _record_calls(payload)

    result = run_benchmark(benchmark, scale=0.0001)
    assert result.iterations >= 1
    assert len(result.samples_ns) >= 1


def test_run_benchmark_rejects_non_positive_scale():
    benchmark = Benchmark(name="x", fn=_noop, iterations=1)
    with pytest.raises(ValueError):
        run_benchmark(benchmark, scale=0)


def test_benchmark_result_statistics_from_known_samples():
    result = BenchmarkResult(
        name="example",
        category="test",
        description="",
        iterations=5,
        inner_loops=1,
        samples_ns=[10, 20, 30, 40, 50],
    )

    assert result.min_ns == 10
    assert result.max_ns == 50
    assert result.median_ns == 30
    assert result.mean_ns == pytest.approx(30.0)
    assert result.p95_ns == pytest.approx(48.0)
    assert result.p99_ns == pytest.approx(49.6)
    # ops_per_second derived from mean_ns of 30ns => ~33.3M/s
    assert result.ops_per_second == pytest.approx(1_000_000_000 / 30)


def test_benchmark_result_inner_loops_divide_samples():
    result = BenchmarkResult(
        name="loops",
        category="test",
        description="",
        iterations=2,
        inner_loops=10,
        samples_ns=[1000, 2000],
    )

    # Each sample divided by inner_loops
    assert result.min_ns == 100
    assert result.max_ns == 200
    assert result.mean_ns == pytest.approx(150.0)


def test_benchmark_result_to_dict_round_trips_through_json():
    result = BenchmarkResult(
        name="dump",
        category="test",
        description="example",
        iterations=3,
        inner_loops=1,
        samples_ns=[100, 200, 300],
    )

    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["name"] == "dump"
    assert payload["iterations"] == 3
    assert payload["ns"]["min"] == 100
    assert payload["ns"]["max"] == 300


def test_benchmark_result_to_dict_uses_standard_json_for_zero_duration_sample():
    result = BenchmarkResult(
        name="zero",
        category="test",
        description="timer returned same tick",
        iterations=1,
        inner_loops=1,
        samples_ns=[0],
    )

    payload = result.to_dict()

    assert payload["ops_per_second"] is None
    json.dumps(payload, allow_nan=False)


def test_format_json_report_uses_standard_json_for_empty_samples():
    metadata = RunMetadata.capture(scale=1.0)
    result = BenchmarkResult(
        name="empty",
        category="test",
        description="synthetic empty measurement",
        iterations=0,
        inner_loops=1,
        samples_ns=[],
    )

    payload = json.loads(format_json_report([result], metadata=metadata))

    assert payload["results"][0]["ops_per_second"] is None
    assert payload["results"][0]["ns"]["median"] is None


def test_format_text_report_groups_by_category():
    metadata = RunMetadata.capture(scale=1.0)
    results = [
        BenchmarkResult(
            name="a.one",
            category="alpha",
            description="",
            iterations=2,
            inner_loops=1,
            samples_ns=[100, 200],
        ),
        BenchmarkResult(
            name="b.one",
            category="beta",
            description="",
            iterations=2,
            inner_loops=1,
            samples_ns=[1000, 2000],
        ),
    ]

    text = format_text_report(results, metadata=metadata)

    assert "[alpha]" in text
    assert "[beta]" in text
    assert "a.one" in text
    assert "b.one" in text


def test_format_json_report_is_valid_json_with_metadata():
    metadata = RunMetadata.capture(scale=0.5)
    results = [
        BenchmarkResult(
            name="x",
            category="cat",
            description="d",
            iterations=1,
            inner_loops=1,
            samples_ns=[10],
        )
    ]

    payload = json.loads(format_json_report(results, metadata=metadata))

    assert payload["metadata"]["scale"] == 0.5
    assert payload["results"][0]["name"] == "x"


def test_discover_benchmarks_returns_unique_names_across_modules():
    benchmarks = runner_module.discover_benchmarks()
    names = [b.name for b in benchmarks]

    assert names, "expected at least one benchmark"
    assert len(names) == len(set(names)), "benchmark names must be unique"


def test_discover_benchmarks_rejects_duplicate_names(monkeypatch, tmp_path):
    # Build two modules-by-name that both yield the same Benchmark name.
    import sys
    import types

    name = "duplicate.bench"
    fake_module = types.ModuleType("benchmarks._test_dup")
    fake_module.BENCHMARKS = [Benchmark(name=name, fn=_noop, iterations=1)]
    sys.modules["benchmarks._test_dup"] = fake_module

    try:
        with pytest.raises(ValueError, match="duplicate benchmark name"):
            runner_module.discover_benchmarks(
                ("benchmarks._test_dup", "benchmarks._test_dup")
            )
    finally:
        sys.modules.pop("benchmarks._test_dup", None)


def test_runner_main_filter_then_list(capsys):
    exit_code = runner_module.main(["--list", "--filter", "^api\\."])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[api]" in captured.out
    assert "api.HybridSearchService.search" in captured.out
    # ui benchmarks should be filtered out by the anchored regex
    assert "[ui]" not in captured.out


def test_runner_main_no_match_returns_two(capsys):
    exit_code = runner_module.main(["--filter", "this-will-never-match"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "No benchmarks selected" in captured.err


def test_runner_main_rejects_invalid_baseline_before_running_benchmarks(
    monkeypatch,
    tmp_path,
    capsys,
):
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{not-json")
    benchmark = Benchmark(name="example", fn=_noop, iterations=1)

    monkeypatch.setattr(runner_module, "discover_benchmarks", lambda: [benchmark])

    def fail_run_benchmark(*_args, **_kwargs):
        raise AssertionError("baseline should be validated before benchmarks run")

    monkeypatch.setattr(runner_module, "run_benchmark", fail_run_benchmark)

    exit_code = runner_module.main(["--baseline", str(baseline), "--quiet"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Invalid benchmark baseline" in captured.err
    assert "valid JSON" in captured.err
