import logging
import math
from typing import Any, Dict, List, Optional, Protocol

from api.search_utils import text_metadata_by_segment_id

logger = logging.getLogger(__name__)

RRF_K = 60


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

    if not vector or any(isinstance(value, bool) for value in vector):
        raise ValueError("query embedding must be a non-empty numeric vector")

    try:
        float_vector = [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise ValueError("query embedding must be a non-empty numeric vector") from exc

    if any(not math.isfinite(value) for value in float_vector):
        raise ValueError("query embedding must be a non-empty numeric vector")

    return float_vector


def _first_id_list(results: Dict[str, Any]) -> List[str]:
    ids = results.get("ids") or [[]]
    if not ids:
        return []
    return ids[0] or []


def _segment_id(doc_id: str, suffix: str) -> str:
    return doc_id.removesuffix(suffix)


def _metadata_string(metadata: Dict[str, Any], field_name: str) -> Optional[str]:
    value = metadata.get(field_name)
    if not isinstance(value, str):
        return None
    return value


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

    if (
        title is None
        or summary is None
        or video_filename is None
        or not video_filename.strip()
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
        query_vector = _query_vector(self.embedding_model, query)

        text_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k * 3,
            where=_where_clause("text", video_filename),
        )
        visual_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k * 3,
            where=_where_clause("visual", video_filename),
        )

        fused_scores: Dict[str, float] = {}
        for rank, doc_id in enumerate(_first_id_list(text_results)):
            segment_id = _segment_id(doc_id, "_text")
            fused_scores[segment_id] = fused_scores.get(segment_id, 0) + 1 / (rank + RRF_K)

        for rank, doc_id in enumerate(_first_id_list(visual_results)):
            segment_id = _segment_id(doc_id, "_visual")
            fused_scores[segment_id] = fused_scores.get(segment_id, 0) + 1 / (rank + RRF_K)

        top_segment_ids = sorted(
            fused_scores,
            key=lambda segment_id: fused_scores[segment_id],
            reverse=True,
        )[:top_k]
        if not top_segment_ids:
            return []

        final_results_data = self.collection.get(
            ids=[f"{segment_id}_text" for segment_id in top_segment_ids],
            include=["metadatas"],
        )
        metadata_by_segment_id = text_metadata_by_segment_id(
            final_results_data.get("ids", []),
            final_results_data.get("metadatas", []),
        )

        formatted_results = []
        for segment_id in top_segment_ids:
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
