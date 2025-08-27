# ingestion_pipeline/utils/metadata_fetcher.py

import os
import logging
from tmdbv3api import TMDb, Movie
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def fetch_movie_metadata(title: str, year: Optional[int] = None) -> Optional[Dict]:
    """
    Fetches movie metadata from The Movie Database (TMDb).
    """
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        logger.error("TMDB_API_KEY environment variable not set. Cannot fetch metadata.")
        return None

    try:
        tmdb = TMDb()
        tmdb.api_key = api_key
        
        movie_api = Movie()
        search_results = movie_api.search(title)
        
        if not search_results:
            logger.warning(f"No movie found on TMDb for title: '{title}'")
            return None

        # Find the best match (often the first result, but filtering by year is better)
        best_match = None
        if year:
            for result in search_results:
                if 'release_date' in result and result.release_date:
                    release_year = int(result.release_date.split('-')[0])
                    if release_year == year:
                        best_match = result
                        break
        
        # If no year match was found or no year was provided, take the first result
        if not best_match:
            best_match = search_results[0]
        
        logger.info(f"Found TMDb match: '{best_match.title}' ({best_match.release_date})")
        
        # Fetch full details for the matched movie
        details = movie_api.details(best_match.id)

        # Format the data into our desired structure
        metadata = {
            "title": details.title,
            "logline": details.overview,
            "genre": ", ".join([genre['name'] for genre in details.genres]),
            "setting": "N/A", # TMDb doesn't have a standard 'setting' field, so we'll leave it
            "main_characters": [] # This is also not a standard field, but we have the title/logline
        }
        return metadata

    except Exception as e:
        logger.error(f"An error occurred while fetching TMDb data: {e}")
        return None
