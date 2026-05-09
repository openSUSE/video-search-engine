"""Unit tests for ``api.demo_bootstrap``.

The bootstrap is a side-effecting integration on the API path, so we
isolate it with a fake collection (capturing upserts) and a fake
embedding model. The Chroma client itself is wrapped through
``core.chroma_setup.get_search_collection``, which tests in
``test_chroma_setup`` already cover; here we only verify the
high-level decision tree.
"""

from __future__ import annotations

import json
import os
import sys
import types
from typing import Any, Dict, List

import pytest

from api import demo_bootstrap


@pytest.fixture(autouse=True)
def _clear_module_cache():
    yield
    # Keep test isolation: the module reads env vars at call time so
    # nothing stateful needs clearing here, but make sure fake
    # ``chromadb`` modules introduced by tests don't leak.
    sys.modules.pop("chromadb", None)


def _install_fake_chromadb(
    monkeypatch: pytest.MonkeyPatch,
    *,
    existing_ids: List[str] | None = None,
    upsert_capture: List[Dict[str, Any]] | None = None,
    heartbeat_raises: Exception | None = None,
):
    fake = types.ModuleType("chromadb")

    class FakeCollection:
        metadata = {"hnsw:space": "cosine"}
        name = "x"

        def get(self, **kwargs):
            return {"ids": list(existing_ids or [])}

        def upsert(self, **kwargs):
            if upsert_capture is not None:
                upsert_capture.append(kwargs)

        def delete(self, **kwargs):
            pass

    class FakeClient:
        def __init__(self, *, host, port):
            self.host = host
            self.port = port

        def heartbeat(self):
            if heartbeat_raises is not None:
                raise heartbeat_raises
            return 1

        def get_or_create_collection(self, name, metadata=None):
            return FakeCollection()

    fake.HttpClient = FakeClient
    monkeypatch.setitem(sys.modules, "chromadb", fake)


def _install_fake_indexing(monkeypatch: pytest.MonkeyPatch, *, return_value: bool = True):
    """Stub out the heavy ``run_indexing`` so we don't need
    sentence-transformers / chromadb upsert under test."""

    captured: Dict[str, Any] = {}

    fake_module = types.ModuleType("ingestion_pipeline.steps.step_04_indexing")

    class _FakeEmbedding:
        def encode(self, texts, **kwargs):
            return [[0.0, 0.0] for _ in texts]

    def fake_create_embedding_model(_config):
        return _FakeEmbedding()

    def fake_run_indexing(**kwargs):
        captured.update(kwargs)
        return return_value

    fake_module.create_embedding_model = fake_create_embedding_model
    fake_module.run_indexing = fake_run_indexing
    monkeypatch.setitem(
        sys.modules, "ingestion_pipeline.steps.step_04_indexing", fake_module
    )
    return captured


@pytest.fixture
def demo_config():
    return {
        "database": {
            "host": "chroma",
            "port": 8000,
            "collection_name": "video_search_engine",
        },
        "models": {"embedding": {"name": "all-MiniLM-L6-v2"}},
        "general": {"device": "cpu"},
    }


def test_bootstrap_disabled_when_env_zero(monkeypatch, demo_config, caplog):
    monkeypatch.setenv("DEMO_BOOTSTRAP", "0")
    _install_fake_chromadb(monkeypatch)
    capture = _install_fake_indexing(monkeypatch)

    with caplog.at_level("INFO", logger="api.demo_bootstrap"):
        demo_bootstrap.bootstrap(demo_config)

    assert not capture, "run_indexing must not be called when DEMO_BOOTSTRAP=0"


def test_bootstrap_skipped_when_collection_already_populated(
    monkeypatch, demo_config
):
    monkeypatch.setenv("DEMO_BOOTSTRAP", "1")
    _install_fake_chromadb(
        monkeypatch,
        existing_ids=["sintel_trailer::segment_0001_text"],
    )
    capture = _install_fake_indexing(monkeypatch)

    demo_bootstrap.bootstrap(demo_config)
    assert not capture, (
        "run_indexing must skip when the demo video already has rows"
    )


def test_bootstrap_runs_indexing_when_collection_empty(monkeypatch, demo_config):
    monkeypatch.setenv("DEMO_BOOTSTRAP", "1")
    monkeypatch.delenv("VIDEO_DATA_PATH", raising=False)
    _install_fake_chromadb(monkeypatch, existing_ids=[])
    capture = _install_fake_indexing(monkeypatch)

    demo_bootstrap.bootstrap(demo_config)

    assert capture, "run_indexing must be called when collection is empty"
    assert capture["video_filename"] == demo_bootstrap.DEMO_VIDEO_FILENAME
    enriched_path = capture["enriched_segments_path"]
    assert enriched_path.endswith(demo_bootstrap.DEMO_ENRICHED_BASENAME)
    # The resolved path should actually exist in the repo so a future
    # refactor that loses the file is caught by this test.
    assert os.path.exists(enriched_path), enriched_path


def test_bootstrap_publishes_video_when_video_dir_writable(
    monkeypatch, demo_config, tmp_path
):
    monkeypatch.setenv("DEMO_BOOTSTRAP", "1")
    monkeypatch.setenv("VIDEO_DATA_PATH", str(tmp_path))
    _install_fake_chromadb(monkeypatch, existing_ids=[])
    _install_fake_indexing(monkeypatch)

    demo_bootstrap.bootstrap(demo_config)

    assert (tmp_path / demo_bootstrap.DEMO_VIDEO_BASENAME).is_file()


def test_bootstrap_skips_when_chroma_unreachable(monkeypatch, demo_config):
    monkeypatch.setenv("DEMO_BOOTSTRAP", "1")
    _install_fake_chromadb(
        monkeypatch, heartbeat_raises=ConnectionError("server down")
    )
    capture = _install_fake_indexing(monkeypatch)

    # Should NOT raise — best-effort bootstrap.
    demo_bootstrap.bootstrap(demo_config)
    assert not capture


def test_bootstrap_handles_missing_enriched_file(monkeypatch, demo_config, tmp_path):
    monkeypatch.setenv("DEMO_BOOTSTRAP", "1")

    # Point _demo_root at an empty directory.
    monkeypatch.setattr(demo_bootstrap, "_demo_root", lambda: str(tmp_path))
    _install_fake_chromadb(monkeypatch, existing_ids=[])
    capture = _install_fake_indexing(monkeypatch)

    demo_bootstrap.bootstrap(demo_config)
    assert not capture
