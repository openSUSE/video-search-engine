# api/main.py

import logging
from fastapi import FastAPI, HTTPException
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from api.schemas import SearchQuery, SearchResponse
from api.search_utils import text_metadata_by_segment_id
from core.config import CONFIG

# --- API SETUP ---
logger = logging.getLogger(__name__)

# Create the FastAPI app instance
app = FastAPI(
    title="Video Search Engine API",
    description="An API for performing semantic search on video segments.",
    version="1.0.0"
)

# --- ML MODEL AND DATABASE LOADING (GLOBAL) ---
# Load models and connect to DB only once when the API starts up.
try:
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer(CONFIG['models']['embedding']['name'], device=CONFIG['general']['device'])
    logger.info("Model loaded successfully.")

    logger.info("Connecting to ChromaDB...")
    db_config = CONFIG['database']
    chroma_client = chromadb.HttpClient(host=db_config['host'], port=db_config['port'])
    collection = chroma_client.get_or_create_collection(
        name=db_config['collection_name'],
        metadata={"hnsw:space": "cosine"}
    )
    logger.info("Successfully connected to ChromaDB collection.")

except Exception as e:
    logger.critical(f"Failed to initialize API dependencies: {e}", exc_info=True)
    # If we can't load models or connect to DB, the API is useless.
    # In a real production system, you might have more graceful error handling.
    raise

# --- HYBRID SEARCH AND RE-RANKING LOGIC ---
def perform_hybrid_search(query: str, top_k: int, video_filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Performs a hybrid search by querying text and visual embeddings separately,
    then re-ranks the results using Reciprocal Rank Fusion (RRF).
    """
    # 1. Create a vector embedding for the user's query
    query_vector = embedding_model.encode(query).tolist()

    # 2. Query both text and visual types in ChromaDB
    # We fetch more results than needed (e.g., 3*top_k) to give the re-ranker more to work with.
    where_clause_text = {"type": "text"}
    where_clause_visual = {"type": "visual"}

    if video_filename:
        where_clause_text = {"$and": [{"type": "text"}, {"video_filename": video_filename}]}
        where_clause_visual = {"$and": [{"type": "visual"}, {"video_filename": video_filename}]}

    text_results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k * 3,
        where=where_clause_text
    )
    
    visual_results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k * 3,
        where=where_clause_visual
    )

    # 3. Re-rank using Reciprocal Rank Fusion (RRF)
    # RRF is a simple but powerful way to combine ranked lists.
    fused_scores = {}
    k = 60  # RRF tuning constant, 60 is a common default.

    # Process text results
    for rank, doc_id in enumerate(text_results['ids'][0]):
        segment_id = doc_id.replace("_text", "")
        if segment_id not in fused_scores:
            fused_scores[segment_id] = 0
        fused_scores[segment_id] += 1 / (rank + k)

    # Process visual results
    for rank, doc_id in enumerate(visual_results['ids'][0]):
        segment_id = doc_id.replace("_visual", "")
        if segment_id not in fused_scores:
            fused_scores[segment_id] = 0
        fused_scores[segment_id] += 1 / (rank + k)

    # 4. Sort segments by their fused score
    reranked_segment_ids = sorted(fused_scores.keys(), key=lambda id: fused_scores[id], reverse=True)

    # 5. Fetch the full metadata for the top-ranked segments
    top_segment_ids = reranked_segment_ids[:top_k]
    if not top_segment_ids:
        return []

    # We only need to fetch the metadata once per segment. Let's fetch the 'text' version.
    final_ids_to_fetch = [f"{id}_text" for id in top_segment_ids]
    final_results_data = collection.get(ids=final_ids_to_fetch, include=["metadatas"])
    metadata_by_segment_id = text_metadata_by_segment_id(
        final_results_data.get("ids", []),
        final_results_data.get("metadatas", []),
    )

    # 6. Format the final results list
    formatted_results = []
    for segment_id in top_segment_ids:
        metadata = metadata_by_segment_id.get(segment_id)
        if metadata:
            formatted_results.append({
                "id": segment_id,
                "score": fused_scores[segment_id],
                **metadata
            })
    
    return formatted_results

# --- API ENDPOINT ---
@app.post("/search", response_model=SearchResponse)
def search(query: SearchQuery):
    """
    Accepts a search query and returns a ranked list of relevant video segments.
    """
    logger.info(f"Received search query: '{query.query}' for video: {query.video_filename}")
    try:
        results = perform_hybrid_search(query.query, query.top_k, query.video_filename)
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
