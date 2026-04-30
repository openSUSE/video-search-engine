import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol

from api.search_utils import text_metadata_by_segment_id

logger = logging.getLogger(__name__)

RRF_K = 60
MAX_SEARCH_LIMIT = 50


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
    encoded = embedding_model.encode(query)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()

    try:
        vector = list(encoded)
    except TypeError as exc:
        raise ValueError("query embedding must be a non-empty numeric vector") from exc

    if not vector:
        raise ValueError("query embedding must be a non-empty numeric vector")

    invalid = ValueError("query embedding must be a non-empty numeric vector")
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
    for doc_id in _first_id_list(results):
        if not isinstance(doc_id, str) or not doc_id.endswith(suffix):
            logger.warning(
                "Skipping malformed search query result id %r; expected suffix %s.",
                doc_id,
                suffix,
            )
            continue
        segment_ids.append(doc_id.removesuffix(suffix))
    return segment_ids


def _metadata_string(metadata: Dict[str, Any], field_name: str) -> Optional[str]:
    value = metadata.get(field_name)
    if not isinstance(value, str):
        return None
    return value.strip()


def _metadata_time(metadata: Dict[str, Any], field_name: str) -> Optional[float]:
    value = metadata.get(field_name)
    if isinstance(value, bool):
        return None
    try:
        time_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(time_value) or time_value < 0:
        return None
    return time_value


def _format_search_result(
    segment_id: str,
    score: float,
    metadata: Any,
) -> Optional[Dict[str, Any]]:
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
        or speakers is None
        or start_time is None
        or end_time is None
        or end_time < start_time
    ):
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


class HybridSearchService:
    def __init__(self, embedding_model: EmbeddingModel, collection: VectorCollection):
        self.embedding_model = embedding_model
        self.collection = collection

    def search(
        self,
        query: str,
        top_k: int,
        video_filename: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        top_k = _search_limit(top_k)
        query_vector = _query_vector(self.embedding_model, query)

        text_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k * 3,
            where=_where_clause("text", video_filename),
            include=[],
        )
        visual_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k * 3,
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
        metadata_by_segment_id = text_metadata_by_segment_id(
            final_results_data.get("ids", []),
            final_results_data.get("metadatas", []),
        )

        formatted_results = []
        for segment_id in ranked_segment_ids:
            metadata = metadata_by_segment_id.get(segment_id)
            result = _format_search_result(
                segment_id,
                fused_scores[segment_id],
                metadata,
            )
            if result is None:
                logger.warning(
                    "Skipping search result for %s with malformed text metadata.",
                    segment_id,
                )
                continue
            formatted_results.append(result)
            if len(formatted_results) == top_k:
                break

        return formatted_results


def create_search_service(config: Dict[str, Any]) -> HybridSearchService:
    import chromadb
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer(
        config["models"]["embedding"]["name"],
        device=config["general"]["device"],
    )
    logger.info("Embedding model loaded successfully.")

    logger.info("Connecting to ChromaDB...")
    db_config = config["database"]
    chroma_client = chromadb.HttpClient(host=db_config["host"], port=db_config["port"])
    collection = chroma_client.get_or_create_collection(
        name=db_config["collection_name"],
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Connected to ChromaDB collection '%s'.", db_config["collection_name"])

    return HybridSearchService(embedding_model, collection)
