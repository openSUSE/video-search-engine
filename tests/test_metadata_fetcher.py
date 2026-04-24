from ingestion_pipeline.utils.metadata_fetcher import fetch_movie_metadata


def test_fetch_movie_metadata_without_api_key_does_not_require_tmdb_sdk(monkeypatch):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)

    assert fetch_movie_metadata("Demo") is None
