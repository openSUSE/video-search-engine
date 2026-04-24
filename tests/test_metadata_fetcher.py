import builtins
import sys
import types

from ingestion_pipeline.utils.metadata_fetcher import fetch_movie_metadata


def test_fetch_movie_metadata_without_api_key_does_not_require_tmdb_sdk(monkeypatch):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)

    assert fetch_movie_metadata("Demo") is None


def test_fetch_movie_metadata_blank_api_key_does_not_require_tmdb_sdk(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", " ")
    imported_tmdb_modules = []
    original_import = builtins.__import__

    def tracking_import(name, *args, **kwargs):
        if name == "tmdbv3api":
            imported_tmdb_modules.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    assert fetch_movie_metadata("Demo") is None
    assert imported_tmdb_modules == []


def test_fetch_movie_metadata_formats_overview_as_synopsis(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")

    fake_module = types.ModuleType("tmdbv3api")

    class FakeTMDb:
        api_key = None

    class FakeSearchResult:
        id = 7
        title = "Demo"
        release_date = "2024-01-01"

        def __contains__(self, key):
            return hasattr(self, key)

    class FakeDetails:
        title = "Demo"
        overview = "Fetched overview."
        genres = [{"name": "Drama"}]

    class FakeMovie:
        def search(self, title):
            return [FakeSearchResult()]

        def details(self, movie_id):
            return FakeDetails()

    fake_module.TMDb = FakeTMDb
    fake_module.Movie = FakeMovie
    monkeypatch.setitem(sys.modules, "tmdbv3api", fake_module)

    metadata = fetch_movie_metadata("Demo", 2024)

    assert metadata["synopsis"] == "Fetched overview."
    assert "logline" not in metadata
