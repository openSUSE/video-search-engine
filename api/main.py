import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Request

from api.schemas import SearchQuery, SearchResponse
from api.search_service import HybridSearchService, create_search_service

logger = logging.getLogger(__name__)


def load_api_config() -> Dict[str, Any]:
    from core.config import CONFIG

    return CONFIG


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.search_service = create_search_service(load_api_config())
    yield


app = FastAPI(
    title="Video Search Engine API",
    description="An API for performing semantic search on video segments.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_search_service(request: Request) -> HybridSearchService:
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Search service is not initialized.")
    return service


@app.post("/search", response_model=SearchResponse)
def search(
    query: SearchQuery,
    search_service: HybridSearchService = Depends(get_search_service),
):
    """
    Accepts a search query and returns a ranked list of relevant video segments.
    """
    logger.info(f"Received search query: '{query.query}' for video: {query.video_filename}")
    try:
        results = search_service.search(query.query, query.top_k, query.video_filename)
        return {"results": results}
    except Exception as e:
        logger.error(f"An error occurred during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during search.")


@app.get("/")
def read_root():
    return {"message": "Video Search Engine API is running. Go to /docs for the API interface."}


@app.get("/healthz")
def healthz():
    return {"ok": True}
