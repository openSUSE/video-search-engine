"""Microbenchmarks for the video search engine.

Each module under this package exports a ``BENCHMARKS`` list of
:class:`benchmarks.harness.Benchmark` instances exercising a hot path of the
production code. The :mod:`benchmarks.runner` module discovers these lists,
executes them with the shared :mod:`benchmarks.harness`, and prints a
human-readable report (or writes a JSON artifact for trend tracking).

The suite is intentionally dependency-light: it only relies on the imports
already pulled in by ``requirements-dev.txt`` plus optional stack imports
that the relevant module already requires.
"""
