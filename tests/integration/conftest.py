"""Integration-test fixtures.

These tests talk to a real ChromaDB instance and skip themselves if the
service isn't reachable. CHROMA_HOST/CHROMA_PORT come from the environment
so the same test file works locally (via `make compose-up`) and in CI
(where the compose-smoke job sets them explicitly).
"""

from __future__ import annotations

import os
import socket
import uuid

import pytest


def _chroma_host() -> str:
    return (os.environ.get("CHROMA_HOST") or "127.0.0.1").strip() or "127.0.0.1"


def _chroma_port() -> int:
    raw = (os.environ.get("CHROMA_PORT") or "8000").strip()
    try:
        return int(raw)
    except ValueError:
        return 8000


def _chroma_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def chroma_endpoint() -> tuple[str, int]:
    host, port = _chroma_host(), _chroma_port()
    if not _chroma_reachable(host, port):
        pytest.skip(
            f"Chroma at {host}:{port} is not reachable; "
            "start it with `make compose-up` before running integration tests."
        )
    return host, port


@pytest.fixture()
def chroma_collection(chroma_endpoint):
    """Yields a uniquely-named Chroma collection and cleans it up afterwards.

    Using a fresh per-test collection name keeps tests isolated even when
    they share the same Chroma instance, which matters in CI where the
    compose-smoke lane runs them serially against one container.
    """
    import chromadb

    host, port = chroma_endpoint
    client = chromadb.HttpClient(host=host, port=port)
    name = f"itest_{uuid.uuid4().hex[:12]}"
    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    try:
        yield collection
    finally:
        try:
            client.delete_collection(name)
        except Exception:
            # If something already deleted the collection (e.g. test failure
            # mid-cleanup), don't mask the original error.
            pass
