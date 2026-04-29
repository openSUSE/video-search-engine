import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Request

from api.schemas import SearchQuery, SearchResponse
from api.search_service import HybridSearchService, create_search_service
from core.logger import setup_logging

logger = logging.getLogger(__name__)


def load_api_config() -> Dict[str, Any]:
    from core.config import CONFIG

    return CONFIG


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Apply LOG_LEVEL before initialising the search service so any startup
    # diagnostics emit at the configured verbosity. setup_logging is
    # idempotent and skips reattachment if uvicorn has already configured a
    # root handler.
    setup_logging()
    try:
        app.state.search_service = create_search_service(load_api_config())
    except Exception:
        logger.exception("Failed to initialize search service during startup.")
        raise
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
    logger.info(
        "Received search query: %r for video: %r",
        query.query,
        query.video_filename,
    )
    started_at = time.monotonic()
    try:
        results = search_service.search(query.query, query.top_k, query.video_filename)
    except Exception:
        duration_ms = (time.monotonic() - started_at) * 1000
        logger.exception(
            "Search failed after %.1fms (top_k=%d, video=%r).",
            duration_ms,
            query.top_k,
            query.video_filename,
        )
        raise HTTPException(status_code=500, detail="Internal server error during search.")
    duration_ms = (time.monotonic() - started_at) * 1000
    logger.info(
        "Search returned %d results in %.1fms (top_k=%d, video=%r)",
        len(results),
        duration_ms,
        query.top_k,
        query.video_filename,
    )
    return {"results": results}


@app.get("/")
def read_root():
    return {
        "message": "Video Search Engine API is running. Go to /docs for the API interface.",
        "version": app.version,
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz(_: HybridSearchService = Depends(get_search_service)):
    return {"ok": True}
