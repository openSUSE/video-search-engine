import builtins
import logging
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


def test_fetch_movie_metadata_skips_results_with_unparsable_release_date(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")

    fake_module = types.ModuleType("tmdbv3api")

    class FakeTMDb:
        api_key = None

    class _Result:
        def __init__(self, id_, title, release_date):
            self.id = id_
            self.title = title
            self.release_date = release_date

        def __contains__(self, key):
            return hasattr(self, key)

    bad = _Result(1, "Bad Date", "TBD")
    good = _Result(2, "Demo (2024)", "2024-01-01")

    class FakeDetails:
        title = "Demo (2024)"
        overview = "The right one."
        genres = [{"name": "Drama"}]

    class FakeMovie:
        def search(self, title):
            return [bad, good]

        def details(self, movie_id):
            assert movie_id == 2
            return FakeDetails()

    fake_module.TMDb = FakeTMDb
    fake_module.Movie = FakeMovie
    monkeypatch.setitem(sys.modules, "tmdbv3api", fake_module)

    metadata = fetch_movie_metadata("Demo", 2024)

    assert metadata is not None
    assert metadata["title"] == "Demo (2024)"


def test_fetch_movie_metadata_year_filter_accepts_attribute_only_results(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")

    fake_module = types.ModuleType("tmdbv3api")

    class FakeTMDb:
        api_key = None

    class _Result:
        def __init__(self, id_, title, release_date):
            self.id = id_
            self.title = title
            self.release_date = release_date

    wrong = _Result(1, "Demo (1999)", "1999-06-30")
    right = _Result(2, "Demo (2024)", "2024-01-01")

    class FakeDetails:
        title = "Demo (2024)"
        overview = "The right release."
        genres = [{"name": "Drama"}]

    class FakeMovie:
        def search(self, title):
            return [wrong, right]

        def details(self, movie_id):
            assert movie_id == 2
            return FakeDetails()

    fake_module.TMDb = FakeTMDb
    fake_module.Movie = FakeMovie
    monkeypatch.setitem(sys.modules, "tmdbv3api", fake_module)

    metadata = fetch_movie_metadata("Demo", 2024)

    assert metadata is not None
    assert metadata["title"] == "Demo (2024)"


def test_fetch_movie_metadata_skips_year_matches_without_ids(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")

    fake_module = types.ModuleType("tmdbv3api")

    class FakeTMDb:
        api_key = None

    class MissingIdResult:
        title = "Demo (2024 missing id)"
        release_date = "2024-01-01"

    class ValidResult:
        id = 2
        title = "Demo (2024)"
        release_date = "2024-02-01"

    class FakeDetails:
        title = "Demo (2024)"
        overview = "The usable release."
        genres = [{"name": "Drama"}]

    class FakeMovie:
        def search(self, title):
            return [MissingIdResult(), ValidResult()]

        def details(self, movie_id):
            assert movie_id == 2
            return FakeDetails()

    fake_module.TMDb = FakeTMDb
    fake_module.Movie = FakeMovie
    monkeypatch.setitem(sys.modules, "tmdbv3api", fake_module)

    metadata = fetch_movie_metadata("Demo", 2024)

    assert metadata is not None
    assert metadata["title"] == "Demo (2024)"


def test_fetch_movie_metadata_warns_when_year_does_not_match_any_result(
    monkeypatch, caplog
):
    monkeypatch.setenv("TMDB_API_KEY", "test-key")

    fake_module = types.ModuleType("tmdbv3api")

    class FakeTMDb:
        api_key = None

    class FakeSearchResult:
        id = 9
        title = "Demo (1999)"
        release_date = "1999-06-30"

        def __contains__(self, key):
            return hasattr(self, key)

    class FakeDetails:
        title = "Demo (1999)"
        overview = "An older release."
        genres = [{"name": "Drama"}]

    class FakeMovie:
        def search(self, title):
            return [FakeSearchResult()]

        def details(self, movie_id):
            return FakeDetails()

    fake_module.TMDb = FakeTMDb
    fake_module.Movie = FakeMovie
    monkeypatch.setitem(sys.modules, "tmdbv3api", fake_module)

    with caplog.at_level(logging.WARNING, logger="ingestion_pipeline.utils.metadata_fetcher"):
        metadata = fetch_movie_metadata("Demo", 2024)

    assert metadata["title"] == "Demo (1999)"
    assert any(
        "year 2024 not found" in record.message and "Demo (1999)" in record.message
        for record in caplog.records
    )
