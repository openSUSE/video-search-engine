# ingestion_pipeline/utils/metadata_fetcher.py

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

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
        from tmdbv3api import TMDb, Movie

        tmdb = TMDb()
        tmdb.api_key = api_key
        
        movie_api = Movie()
        search_results = movie_api.search(title)
        
        if not search_results:
            logger.warning("No movie found on TMDb for title: %r", title)
            return None

        # Find the best match (often the first result, but filtering by year is better)
        best_match = None
        if year:
            for result in search_results:
                if 'release_date' in result and result.release_date:
                    try:
                        release_year = int(result.release_date.split('-')[0])
                    except ValueError:
                        # Some TMDb rows return non-numeric prefixes ("TBD", "Q1 2024");
                        # skip them rather than failing the whole fetch.
                        continue
                    if release_year == year:
                        best_match = result
                        break

        # If no year match was found or no year was provided, take the first result
        if not best_match:
            fallback = search_results[0]
            if year:
                logger.warning(
                    "TMDb year %s not found for title %r; using first result %r (%s).",
                    year,
                    title,
                    getattr(fallback, "title", "?"),
                    getattr(fallback, "release_date", "?"),
                )
            best_match = fallback

        logger.info("Found TMDb match: %r (%s)", best_match.title, best_match.release_date)
        
        # Fetch full details for the matched movie
        details = movie_api.details(best_match.id)

        # Format the data into our desired structure
        metadata = {
            "title": details.title,
            "synopsis": details.overview,
            "genre": ", ".join([genre['name'] for genre in details.genres]),
        }
        return metadata

    except Exception:
        logger.exception("An error occurred while fetching TMDb data.")
        return None
