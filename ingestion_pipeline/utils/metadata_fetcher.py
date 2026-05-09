# ingestion_pipeline/utils/metadata_fetcher.py

import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# One quick retry on rate-limit/transient TMDb errors. We deliberately keep
# this small: the previous bare except Exception swallowed every failure as
# "no movie found", masking persistent rate limits as if the title didn't
# exist.
_TMDB_RETRY_ATTEMPTS = 2
_TMDB_RETRY_DELAY_SECONDS = 1.0


def _metadata_field(item: Any, field_name: str) -> Any:
    if isinstance(item, dict):
        return item.get(field_name)
    return getattr(item, field_name, None)


def _release_year(result: Any) -> Optional[int]:
    release_date = _metadata_field(result, "release_date")
    if not release_date:
        return None

    try:
        return int(str(release_date).split("-")[0])
    except ValueError:
        # Some TMDb rows return non-numeric prefixes ("TBD", "Q1 2024");
        # skip them rather than failing the whole fetch.
        return None


def _metadata_id(result: Any) -> Optional[int]:
    movie_id = _metadata_field(result, "id")
    if isinstance(movie_id, bool):
        return None
    if isinstance(movie_id, int) and movie_id > 0:
        return movie_id
    if isinstance(movie_id, str):
        movie_id = movie_id.strip()
        if movie_id.isdecimal():
            parsed_id = int(movie_id)
            if parsed_id > 0:
                return parsed_id
    return None


def _genre_names(details: Any) -> str:
    names = []
    for genre in _metadata_field(details, "genres") or []:
        genre_name = _metadata_field(genre, "name")
        if isinstance(genre_name, str) and genre_name.strip():
            names.append(genre_name.strip())
    return ", ".join(names)


def fetch_movie_metadata(title: str, year: Optional[int] = None) -> Optional[Dict]:
    """
    Fetches movie metadata from The Movie Database (TMDb).
    """
    api_key = (os.getenv("TMDB_API_KEY") or "").strip()
    if not api_key:
        logger.warning(
            "TMDB_API_KEY is not set; skipping TMDb metadata lookup."
        )
        return None

    try:
        import requests
        from tmdbv3api import TMDb, Movie
    except ImportError as exc:
        logger.warning("TMDb client not available: %s", exc)
        return None
    try:
        from tmdbv3api.exceptions import TMDbException
    except ImportError:
        # Older tmdbv3api versions and some test stubs ship without this
        # submodule. Fall back to RuntimeError so the except chain below
        # still has a concrete (if broader) handle.
        TMDbException = RuntimeError  # type: ignore[assignment]

    tmdb = TMDb()
    tmdb.api_key = api_key
    movie_api = Movie()

    last_error: Optional[BaseException] = None
    for attempt in range(1, _TMDB_RETRY_ATTEMPTS + 1):
        try:
            search_results = movie_api.search(title)

            if not search_results:
                logger.warning("No movie found on TMDb for title: %r", title)
                return None

            usable_results = [
                result for result in search_results if _metadata_id(result) is not None
            ]
            if not usable_results:
                logger.warning(
                    "TMDb search returned no usable movie IDs for title: %r",
                    title,
                )
                return None

            best_match = None
            if year:
                for result in usable_results:
                    if _release_year(result) == year:
                        best_match = result
                        break

            if not best_match:
                fallback = usable_results[0]
                if year:
                    logger.warning(
                        "TMDb year %s not found for title %r; using first result %r (%s).",
                        year,
                        title,
                        _metadata_field(fallback, "title") or "?",
                        _metadata_field(fallback, "release_date") or "?",
                    )
                best_match = fallback

            logger.info(
                "Found TMDb match: %r (%s)",
                _metadata_field(best_match, "title"),
                _metadata_field(best_match, "release_date"),
            )

            details = movie_api.details(_metadata_id(best_match))

            return {
                "title": _metadata_field(details, "title"),
                "synopsis": _metadata_field(details, "overview"),
                "genre": _genre_names(details),
            }
        except (requests.RequestException, TMDbException) as exc:
            last_error = exc
            if attempt < _TMDB_RETRY_ATTEMPTS:
                logger.warning(
                    "TMDb request failed (attempt %d/%d) for %r: %s; retrying.",
                    attempt,
                    _TMDB_RETRY_ATTEMPTS,
                    title,
                    exc,
                )
                time.sleep(_TMDB_RETRY_DELAY_SECONDS)
                continue
            logger.warning(
                "TMDb request failed after %d attempts for %r: %s",
                _TMDB_RETRY_ATTEMPTS,
                title,
                exc,
            )
            return None
        except (ValueError, KeyError) as exc:
            logger.warning("TMDb response parse error for %r: %s", title, exc)
            return None

    logger.warning("TMDb fetch exhausted retries for %r: %s", title, last_error)
    return None
