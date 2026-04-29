"""Filesystem helpers shared across packages.

Workloads that rewrite a JSON file in place (enrichment progress, speaker maps)
need the destination to be either fully updated or unchanged across a crash —
``open(path, 'w')`` truncates the file before serialization runs, so a SIGKILL
or container OOM mid-write erases prior work. ``atomic_write_json`` delegates
the durability bit to ``os.replace``, which is atomic on POSIX and Windows.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Union


PathLike = Union[str, os.PathLike[str]]


def atomic_write_json(path: PathLike, data: Any, indent: int = 2) -> None:
    target = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(target))
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(target) + ".",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
