"""Lightweight timing harness for the video-search-engine benchmark suite.

The harness is intentionally implemented on top of ``time.perf_counter_ns`` and
``statistics`` so that it does not add any third-party runtime dependency. It
mirrors the behaviour of well-known microbenchmark tools at a much smaller
scope:

* ``Benchmark`` describes a single measurement: a callable, an optional setup
  callable, the number of iterations, and an optional ``inner_loops`` factor
  used when the callable performs N units of work per call.
* ``run_benchmark`` executes a benchmark with a configurable warm-up, returns a
  :class:`BenchmarkResult` with summary statistics expressed in nanoseconds and
  derived ops-per-second values.
* ``BenchmarkResult`` is JSON-serialisable so callers can persist reports for
  trend tracking in CI.

The harness deliberately avoids forcing a single global iteration count: each
benchmark declares the iteration count that keeps it both stable (low jitter)
and quick (sub-second on a developer laptop). The runner exposes a
``--scale`` knob to multiply or divide every benchmark uniformly so the same
suite can run as a smoke test in CI or as a longer offline measurement.
"""

from __future__ import annotations

import gc
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


SetupFn = Callable[[], Any]
"""Setup callables receive no arguments and return the value passed to ``fn``."""

BenchmarkFn = Callable[[Any], Any]
"""Benchmark callables receive the value returned by ``setup`` (or ``None``)."""


@dataclass(frozen=True)
class Benchmark:
    """Describes a single measurement.

    Attributes:
        name: Human-readable identifier. Should be unique across the suite.
        fn: Callable executed once per timed iteration. It receives the value
            returned by ``setup`` (or ``None`` when ``setup`` is omitted) so
            that fixtures can be prepared outside the timing window.
        setup: Optional callable invoked once before timing begins. Use this to
            build expensive fixtures (large fake collections, parsed payloads,
            ...). The returned value is passed verbatim to ``fn``.
        iterations: Number of timed iterations. Pick a value that keeps the
            benchmark below ~250ms total wall time on a developer laptop while
            still staying above 100µs per iteration to keep jitter low.
        warmup: Number of untimed warm-up iterations executed before timing
            begins. Two is enough for most CPython hot paths.
        inner_loops: Number of "logical" units of work performed inside ``fn``
            per call. ``run_benchmark`` divides each measurement by this value
            to express timings per logical operation. Defaults to 1.
        category: Free-form grouping label (e.g. ``"api"``, ``"ingestion"``)
            used by the runner to organise the printed report.
        description: Short blurb explaining what the benchmark proves. Shown in
            verbose runner output and embedded in JSON reports.
    """

    name: str
    fn: BenchmarkFn
    setup: Optional[SetupFn] = None
    iterations: int = 200
    warmup: int = 2
    inner_loops: int = 1
    category: str = "misc"
    description: str = ""


@dataclass
class BenchmarkResult:
    """Summary statistics for a single benchmark run."""

    name: str
    category: str
    description: str
    iterations: int
    inner_loops: int
    samples_ns: List[int] = field(default_factory=list)

    @property
    def total_ns(self) -> int:
        return sum(self.samples_ns)

    @property
    def per_op_ns(self) -> List[float]:
        if self.inner_loops <= 1:
            return [float(value) for value in self.samples_ns]
        loops = float(self.inner_loops)
        return [value / loops for value in self.samples_ns]

    def _percentile(self, fraction: float) -> float:
        values = sorted(self.per_op_ns)
        if not values:
            return float("nan")
        if len(values) == 1:
            return values[0]
        # Linear interpolation matching numpy.percentile(..., method="linear").
        position = fraction * (len(values) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    @property
    def mean_ns(self) -> float:
        values = self.per_op_ns
        return statistics.fmean(values) if values else float("nan")

    @property
    def stdev_ns(self) -> float:
        values = self.per_op_ns
        if len(values) < 2:
            return 0.0
        return statistics.stdev(values)

    @property
    def min_ns(self) -> float:
        values = self.per_op_ns
        return min(values) if values else float("nan")

    @property
    def max_ns(self) -> float:
        values = self.per_op_ns
        return max(values) if values else float("nan")

    @property
    def median_ns(self) -> float:
        return self._percentile(0.5)

    @property
    def p95_ns(self) -> float:
        return self._percentile(0.95)

    @property
    def p99_ns(self) -> float:
        return self._percentile(0.99)

    @property
    def ops_per_second(self) -> float:
        mean = self.mean_ns
        if not math.isfinite(mean) or mean <= 0:
            return float("inf")
        return 1_000_000_000.0 / mean

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "iterations": self.iterations,
            "inner_loops": self.inner_loops,
            "ops_per_second": self.ops_per_second,
            "ns": {
                "min": self.min_ns,
                "median": self.median_ns,
                "mean": self.mean_ns,
                "p95": self.p95_ns,
                "p99": self.p99_ns,
                "max": self.max_ns,
                "stdev": self.stdev_ns,
                "total": self.total_ns,
            },
        }


def _scaled_iterations(benchmark: Benchmark, scale: float) -> int:
    if scale <= 0:
        raise ValueError("scale must be positive")
    scaled = max(1, int(round(benchmark.iterations * scale)))
    return scaled


def run_benchmark(benchmark: Benchmark, *, scale: float = 1.0) -> BenchmarkResult:
    """Execute ``benchmark`` and return a :class:`BenchmarkResult`.

    ``scale`` multiplies the iteration count uniformly. Use values below 1.0
    for smoke tests and values above 1.0 for offline measurement. The garbage
    collector is disabled around timed iterations to keep variance low and
    re-enabled afterwards regardless of failure mode.
    """

    fixture = benchmark.setup() if benchmark.setup else None
    fn = benchmark.fn
    iterations = _scaled_iterations(benchmark, scale)
    warmup = max(0, benchmark.warmup)

    for _ in range(warmup):
        fn(fixture)

    samples: List[int] = []
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        for _ in range(iterations):
            start = time.perf_counter_ns()
            fn(fixture)
            samples.append(time.perf_counter_ns() - start)
    finally:
        if gc_was_enabled:
            gc.enable()

    return BenchmarkResult(
        name=benchmark.name,
        category=benchmark.category,
        description=benchmark.description,
        iterations=iterations,
        inner_loops=benchmark.inner_loops,
        samples_ns=samples,
    )
