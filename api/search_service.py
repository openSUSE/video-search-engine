import logging
import math
from collections.abc import Mapping
from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol

from api.search_utils import (
    is_usable_segment_id,
    is_usable_video_filename_scope,
    text_metadata_by_segment_id,
)

logger = logging.getLogger(__name__)

RRF_K = 60
MAX_SEARCH_LIMIT = 50
MAX_QUERY_LENGTH = 1000
MAX_VIDEO_FILENAME_LENGTH = 512
# Per-modality candidate pool before reciprocal-rank fusion. The previous
# top_k*3 (=15 for default top_k=5) dropped many co-occurring text+visual
# hits before reranking; widening the pool meaningfully improves recall on
# multi-modal queries without changing top_k semantics.
CANDIDATE_POOL_FLOOR = 50
CANDIDATE_POOL_MULTIPLIER = 5


class EmbeddingModel(Protocol):
    def encode(self, query: str) -> Any:
        ...


class VectorCollection(Protocol):
    def query(self, **kwargs: Any) -> Dict[str, Any]:
        ...

    def get(self, **kwargs: Any) -> Dict[str, Any]:
        ...


def _where_clause(doc_type: str, video_filename: Optional[str]) -> Dict[str, Any]:
    if video_filename:
        return {"$and": [{"type": doc_type}, {"video_filename": video_filename}]}
    return {"type": doc_type}


def _query_vector(embedding_model: EmbeddingModel, query: str) -> List[float]:
    # normalize_embeddings=True so query vectors live on the unit sphere,
    # matching the indexing-side encode and giving cosine its cleanest form.
    try:
        encoded = embedding_model.encode(query, normalize_embeddings=True)
    except TypeError:
        # Allow Protocol-conforming stubs in tests that don't accept the kwarg.
        encoded = embedding_model.encode(query)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()

    invalid = ValueError("query embedding must be a non-empty numeric vector")
    if isinstance(encoded, (str, bytes, bytearray, Mapping)):
        raise invalid

    try:
        vector = list(encoded)
    except TypeError as exc:
        raise invalid from exc

    if not vector:
        raise invalid

    float_vector: List[float] = []
    for value in vector:
        if isinstance(value, bool):
            raise invalid
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise invalid from exc
        if not math.isfinite(number):
            raise invalid
        float_vector.append(number)

    return float_vector


def _search_limit(top_k: Any) -> int:
    if type(top_k) is not int or not 1 <= top_k <= MAX_SEARCH_LIMIT:
        raise ValueError(f"top_k must be an integer between 1 and {MAX_SEARCH_LIMIT}")
    return top_k


def _search_text(query: Any) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a non-empty string")

    query = query.strip()
    if not query or len(query) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"query must be a non-empty string up to {MAX_QUERY_LENGTH} characters"
        )
    return query


def _duration_bound(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a non-negative number")
    bound = float(value)
    if not math.isfinite(bound) or bound < 0:
        raise ValueError(f"{field_name} must be a finite, non-negative number")
    return bound


def _video_filename_filter(video_filename: Any) -> Optional[str]:
    if video_filename is None:
        return None
    if not isinstance(video_filename, str):
        raise ValueError("video_filename must be a non-empty string when provided")

    video_filename = video_filename.strip()
    if not video_filename or len(video_filename) > MAX_VIDEO_FILENAME_LENGTH:
        raise ValueError(
            "video_filename must be a non-empty string up to "
            f"{MAX_VIDEO_FILENAME_LENGTH} characters"
        )
    if not is_usable_video_filename_scope(video_filename):
        raise ValueError("video_filename must be a filename")
    return video_filename


def _first_id_list(results: Dict[str, Any]) -> List[Any]:
    if not isinstance(results, dict):
        logger.warning("Skipping malformed search query results: expected a mapping.")
        return []

    ids = results.get("ids") or []
    if not ids:
        return []
    if not isinstance(ids, list):
        logger.warning("Skipping malformed search query results: ids must be a list.")
        return []

    first_ids = ids[0]
    if not isinstance(first_ids, (list, tuple)):
        logger.warning(
            "Skipping malformed search query results: first ids row must be a list."
        )
        return []
    return list(first_ids)


def _query_segment_ids(results: Dict[str, Any], suffix: str) -> List[str]:
    segment_ids = []
    seen_segment_ids = set()
    for doc_id in _first_id_list(results):
        if not isinstance(doc_id, str) or not doc_id.endswith(suffix):
            logger.warning(
                "Skipping malformed search query result id %r; expected suffix %s.",
                doc_id,
                suffix,
            )
            continue
        segment_id = doc_id.removesuffix(suffix)
        if not is_usable_segment_id(segment_id):
            logger.warning(
                "Skipping malformed search query result id %r; segment id is not "
                "usable.",
                doc_id,
            )
            continue
        if segment_id in seen_segment_ids:
            continue
        seen_segment_ids.add(segment_id)
        segment_ids.append(segment_id)
    return segment_ids


def _metadata_string(metadata: Dict[str, Any], field_name: str) -> Optional[str]:
    value = metadata.get(field_name)
    if not isinstance(value, str):
        return None
    return value.strip()


def _metadata_time(metadata: Dict[str, Any], field_name: str) -> Optional[float]:
    value = metadata.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    time_value = float(value)
    if not math.isfinite(time_value) or time_value < 0:
        return None
    return time_value


def _scoped_video_filename(segment_id: str) -> Optional[str]:
    if "::" not in segment_id:
        return None

    video_filename, _segment_name = segment_id.rsplit("::", 1)
    video_filename = video_filename.strip()
    return video_filename or None


def _format_search_result(
    segment_id: str,
    score: float,
    metadata: Any,
    requested_video_filename: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not is_usable_segment_id(segment_id):
        return None
    if not isinstance(metadata, dict):
        return None

    title = _metadata_string(metadata, "title")
    summary = _metadata_string(metadata, "summary")
    video_filename = _metadata_string(metadata, "video_filename")
    speakers = _metadata_string(metadata, "speakers")
    start_time = _metadata_time(metadata, "start_time")
    end_time = _metadata_time(metadata, "end_time")

    # _metadata_string already strips, so falsy means missing or
    # whitespace-only. speakers is checked with `is None` (not `not`) so an
    # empty string passes through — that's the legitimate "no speakers
    # identified" case the UI renders as "N/A".
    if (
        not title
        or not summary
        or not video_filename
        or not is_usable_video_filename_scope(video_filename)
        or speakers is None
        or start_time is None
        or end_time is None
        or end_time < start_time
    ):
        return None

    segment_video_filename = _scoped_video_filename(segment_id)
    if requested_video_filename and video_filename != requested_video_filename:
        return None
    if segment_video_filename and video_filename != segment_video_filename:
        return None

    return {
        "id": segment_id,
        "score": score,
        "title": title,
        "summary": summary,
        "start_time": start_time,
        "end_time": end_time,
        "video_filename": video_filename,
        "speakers": speakers,
    }


def _fetched_text_metadata_by_segment_id(results: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(results, dict):
        logger.warning(
            "Skipping malformed search metadata fetch results: expected a mapping."
        )
        return {}

    return text_metadata_by_segment_id(
        results.get("ids", []),
        results.get("metadatas", []),
    )


class HybridSearchService:
    def __init__(self, embedding_model: EmbeddingModel, collection: VectorCollection):
        self.embedding_model = embedding_model
        self.collection = collection

    def search(
        self,
        query: str,
        top_k: int,
        video_filename: Optional[str] = None,
        min_duration_sec: Optional[float] = None,
        max_duration_sec: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        top_k = _search_limit(top_k)
        query = _search_text(query)
        video_filename = _video_filename_filter(video_filename)
        min_duration_sec = _duration_bound(min_duration_sec, "min_duration_sec")
        max_duration_sec = _duration_bound(max_duration_sec, "max_duration_sec")
        query_vector = _query_vector(self.embedding_model, query)

        candidate_pool = max(top_k * CANDIDATE_POOL_MULTIPLIER, CANDIDATE_POOL_FLOOR)
        text_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=candidate_pool,
            where=_where_clause("text", video_filename),
            include=[],
        )
        visual_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=candidate_pool,
            where=_where_clause("visual", video_filename),
            include=[],
        )

        fused_scores: Dict[str, float] = defaultdict(float)
        for results, suffix in ((text_results, "_text"), (visual_results, "_visual")):
            for rank, segment_id in enumerate(_query_segment_ids(results, suffix)):
                fused_scores[segment_id] += 1 / (rank + RRF_K)

        ranked_segment_ids = sorted(
            fused_scores, key=fused_scores.__getitem__, reverse=True
        )
        if not ranked_segment_ids:
            return []

        final_results_data = self.collection.get(
            ids=[f"{segment_id}_text" for segment_id in ranked_segment_ids],
            include=["metadatas"],
        )
        metadata_by_segment_id = _fetched_text_metadata_by_segment_id(final_results_data)

        formatted_results = []
        for segment_id in ranked_segment_ids:
            metadata = metadata_by_segment_id.get(segment_id)
            result = _format_search_result(
                segment_id,
                fused_scores[segment_id],
                metadata,
                requested_video_filename=video_filename,
            )
            if result is None:
                logger.warning(
                    "Skipping search result for %s with malformed text metadata.",
                    segment_id,
                )
                continue
            duration = result["end_time"] - result["start_time"]
            if min_duration_sec is not None and duration < min_duration_sec:
                continue
            if max_duration_sec is not None and duration > max_duration_sec:
                continue
            formatted_results.append(result)
            if len(formatted_results) == top_k:
                break

        return formatted_results


CHROMA_CONNECT_MAX_ATTEMPTS = 6
CHROMA_CONNECT_BACKOFF_BASE_SECONDS = 1.0
CHROMA_CONNECT_BACKOFF_MAX_SECONDS = 8.0


def create_search_service(config: Dict[str, Any]) -> HybridSearchService:
    """Build the hybrid search service.

    The embedding model loads first and unconditionally; if Chroma is slow
    to come up we still want the API process alive so /readyz can return a
    503 instead of the pod crashlooping. The Chroma connect attempts have a
    bounded retry rather than the unbounded blocking call SentenceTransformer
    + chromadb.HttpClient previously combined for.
    """
    import time

    import chromadb
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer(
        config["models"]["embedding"]["name"],
        device=config["general"]["device"],
    )
    logger.info("Embedding model loaded successfully.")

    from core.chroma_setup import get_search_collection

    db_config = config["database"]
    last_error: Optional[BaseException] = None
    for attempt in range(1, CHROMA_CONNECT_MAX_ATTEMPTS + 1):
        try:
            logger.info(
                "Connecting to ChromaDB (attempt %d/%d)...",
                attempt,
                CHROMA_CONNECT_MAX_ATTEMPTS,
            )
            chroma_client = chromadb.HttpClient(
                host=db_config["host"],
                port=db_config["port"],
            )
            # Chroma's HttpClient is lazy; force a real round-trip so a
            # broken connect surfaces here, not on the first request.
            chroma_client.heartbeat()
            collection = get_search_collection(
                chroma_client, db_config["collection_name"]
            )
            logger.info(
                "Connected to ChromaDB collection '%s'.",
                db_config["collection_name"],
            )
            return HybridSearchService(embedding_model, collection)
        except Exception as exc:
            last_error = exc
            if attempt == CHROMA_CONNECT_MAX_ATTEMPTS:
                break
            backoff = min(
                CHROMA_CONNECT_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                CHROMA_CONNECT_BACKOFF_MAX_SECONDS,
            )
            logger.warning(
                "Chroma connect failed (%s); retrying in %.1fs.",
                exc,
                backoff,
            )
            time.sleep(backoff)

    raise RuntimeError(
        f"Chroma connect failed after {CHROMA_CONNECT_MAX_ATTEMPTS} attempts: {last_error}"
    )
