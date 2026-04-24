import sys
from types import SimpleNamespace

import inspect_db


def test_inspect_collection_loads_config_without_path_argument(monkeypatch, capsys):
    calls = {}

    def fake_load_config():
        calls["loaded"] = True
        return {
            "database": {
                "host": "localhost",
                "port": 8000,
                "collection_name": "video_search_engine",
            }
        }

    class FakeCollection:
        def count(self):
            return 0

    class FakeClient:
        def __init__(self, host, port):
            calls["client"] = {"host": host, "port": port}

        def get_collection(self, name):
            calls["collection"] = name
            return FakeCollection()

    monkeypatch.setattr(inspect_db, "_load_config", fake_load_config)
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(HttpClient=FakeClient))

    inspect_db.inspect_collection(fetch_limit=3)

    assert calls["loaded"] is True
    assert calls["client"] == {"host": "localhost", "port": 8000}
    assert calls["collection"] == "video_search_engine"
    assert "The collection is empty." in capsys.readouterr().out
