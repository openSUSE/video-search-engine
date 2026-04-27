"""Command-line runner for the benchmark suite.

Discovery is intentionally explicit: each benchmark module exports a
``BENCHMARKS`` list. Adding a module requires importing it from
``BENCHMARK_MODULES`` below — that single registry is also the canonical
source of truth for ``--list``.

Usage::

    python -m benchmarks.runner                  # full suite, text report
    python -m benchmarks.runner --filter api.    # only API-prefixed names
    python -m benchmarks.runner --json out.json  # additional JSON artifact
    python -m benchmarks.runner --scale 0.25     # smoke run for CI
    python -m benchmarks.runner --list           # print available benchmarks
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path
from typing import Iterable, List

# Ensure the project root is importable when running as ``python -m`` from a
# checkout that hasn't been ``pip install -e``'d.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.harness import Benchmark, BenchmarkResult, run_benchmark  # noqa: E402
from benchmarks.reporters import (  # noqa: E402
    RunMetadata,
    format_json_report,
    format_text_report,
)


BENCHMARK_MODULES = (
    "benchmarks.bench_config",
    "benchmarks.bench_api",
    "benchmarks.bench_segmentation",
    "benchmarks.bench_enrichment",
    "benchmarks.bench_indexing",
    "benchmarks.bench_jobs",
    "benchmarks.bench_ui",
)


def discover_benchmarks(modules: Iterable[str] = BENCHMARK_MODULES) -> List[Benchmark]:
    """Import every module in ``modules`` and collect their ``BENCHMARKS`` list."""

    discovered: List[Benchmark] = []
    seen_names: set[str] = set()
    for module_name in modules:
        module = importlib.import_module(module_name)
        for benchmark in getattr(module, "BENCHMARKS", ()):  # type: ignore[union-attr]
            if benchmark.name in seen_names:
                raise ValueError(
                    f"duplicate benchmark name: {benchmark.name} in {module_name}"
                )
            seen_names.add(benchmark.name)
            discovered.append(benchmark)
    return discovered


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.runner",
        description="Run the video-search-engine benchmark suite.",
    )
    parser.add_argument(
        "--filter",
        dest="patterns",
        action="append",
        default=[],
        help=(
            "Regular expression matched against benchmark name and category. "
            "May be passed multiple times — benchmarks matching ANY pattern run."
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to the iteration count of every benchmark. "
            "Use 0.1-0.25 for smoke runs in CI, 2-5 for offline runs."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Write a JSON report to this path in addition to the text report.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the discovered benchmarks and exit.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-benchmark progress lines on stderr.",
    )
    return parser.parse_args(argv)


def _benchmark_matches(benchmark: Benchmark, patterns: List[re.Pattern[str]]) -> bool:
    if not patterns:
        return True
    # Match against category and name independently so that anchors like
    # ``^api\.`` Just Work without forcing callers to know the internal
    # ``category::name`` separator.
    return any(
        pattern.search(benchmark.name) or pattern.search(benchmark.category)
        for pattern in patterns
    )


def _print_listing(benchmarks: Iterable[Benchmark]) -> None:
    grouped: dict[str, List[Benchmark]] = {}
    for benchmark in benchmarks:
        grouped.setdefault(benchmark.category, []).append(benchmark)
    for category in sorted(grouped):
        print(f"[{category}]")
        for benchmark in sorted(grouped[category], key=lambda b: b.name):
            print(f"  - {benchmark.name} (iters={benchmark.iterations})")
            if benchmark.description:
                print(f"      {benchmark.description}")


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    benchmarks = discover_benchmarks()

    patterns = [re.compile(pattern) for pattern in args.patterns]
    selected = [b for b in benchmarks if _benchmark_matches(b, patterns)]

    if args.list:
        _print_listing(selected)
        return 0

    if not selected:
        print("No benchmarks selected. Use --list to see available names.", file=sys.stderr)
        return 2

    results: List[BenchmarkResult] = []
    for index, benchmark in enumerate(selected, start=1):
        if not args.quiet:
            print(
                f"[{index:>2d}/{len(selected)}] {benchmark.name} ...",
                file=sys.stderr,
                end="",
                flush=True,
            )
        result = run_benchmark(benchmark, scale=args.scale)
        results.append(result)
        if not args.quiet:
            print(
                f" mean={result.mean_ns / 1_000:.2f}µs"
                f" p95={result.p95_ns / 1_000:.2f}µs",
                file=sys.stderr,
                flush=True,
            )

    metadata = RunMetadata.capture(scale=args.scale)
    sys.stdout.write(format_text_report(results, metadata=metadata))

    if args.json_path:
        Path(args.json_path).write_text(
            format_json_report(results, metadata=metadata)
        )
        if not args.quiet:
            print(f"Wrote JSON report to {args.json_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
