import json

import pytest

from core.atomic_io import atomic_write_json


def test_atomic_write_json_replaces_existing_file_on_success(tmp_path):
    output_path = tmp_path / "data.json"
    output_path.write_text(json.dumps([{"segment_id": "old"}]))

    atomic_write_json(str(output_path), [{"segment_id": "new"}])

    assert json.loads(output_path.read_text()) == [{"segment_id": "new"}]
    leftover = sorted(p.name for p in tmp_path.iterdir())
    assert leftover == ["data.json"]


def test_atomic_write_json_creates_file_when_missing(tmp_path):
    output_path = tmp_path / "fresh.json"

    atomic_write_json(str(output_path), {"hello": "world"})

    assert json.loads(output_path.read_text()) == {"hello": "world"}


def test_atomic_write_json_keeps_existing_file_when_serialization_fails(tmp_path):
    output_path = tmp_path / "data.json"
    output_path.write_text(json.dumps([{"segment_id": "previous"}]))

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        atomic_write_json(str(output_path), {"value": Unserializable()})

    # Existing content is untouched, and no temp file leftover.
    assert json.loads(output_path.read_text()) == [{"segment_id": "previous"}]
    leftover = sorted(p.name for p in tmp_path.iterdir())
    assert leftover == ["data.json"]


def test_atomic_write_json_rejects_nonfinite_numbers_without_replacing_file(tmp_path):
    output_path = tmp_path / "data.json"
    output_path.write_text(json.dumps({"value": "previous"}))

    with pytest.raises(ValueError):
        atomic_write_json(str(output_path), {"value": float("nan")})

    assert json.loads(output_path.read_text()) == {"value": "previous"}
    leftover = sorted(p.name for p in tmp_path.iterdir())
    assert leftover == ["data.json"]


def test_atomic_write_json_honors_indent(tmp_path):
    output_path = tmp_path / "data.json"

    atomic_write_json(str(output_path), {"a": 1}, indent=4)

    assert output_path.read_text() == '{\n    "a": 1\n}'


def test_atomic_write_json_raises_when_parent_directory_missing(tmp_path):
    # Locks in the deliberate fail-fast behavior — callers are expected to
    # ensure the parent directory exists before invoking atomic_write_json.
    # Silently mkdir-ing parents here would mask typo'd paths.
    missing = tmp_path / "does_not_exist" / "data.json"

    with pytest.raises(FileNotFoundError):
        atomic_write_json(str(missing), {"k": 1})


def test_atomic_write_json_accepts_pathlike(tmp_path):
    output_path = tmp_path / "data.json"

    # Pass a pathlib.Path directly; no str() coercion at the call site.
    atomic_write_json(output_path, {"k": "v"})

    assert json.loads(output_path.read_text()) == {"k": "v"}
