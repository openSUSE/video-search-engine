# ingestion_pipeline/steps/step_04_indexing.py

import json
import logging
import math
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

DOCUMENT_ID_SCOPE_DELIMITER = "::"


class EmbeddingModel(Protocol):
    def encode(self, sentences: List[str], **kwargs: Any) -> Any:
        ...


class VectorCollection(Protocol):
    def upsert(self, **kwargs: Any) -> Any:
        ...

    def get(self, **kwargs: Any) -> Dict[str, Any]:
        ...

    def delete(self, **kwargs: Any) -> Any:
        ...


def _join_metadata_values(values: Any) -> str:
    if not values:
        return ""
    if isinstance(values, str):
        return values.strip()

    cleaned = [str(value).strip() for value in values]
    return ", ".join(value for value in cleaned if value)


def _join_embedding_parts(parts: List[str]) -> str:
    """Strip each part and join non-blank results with ". " for embedding input.

    Strips per-part because the validators only require list-of-string and
    don't reject blank or whitespace-only entries (e.g. an LLM keyword like
    " " or an empty caption row). Without the strip, those would surface
    as "..." artifacts in the embedding text.
    """
    cleaned = [str(part).strip() for part in parts]
    return ". ".join(part for part in cleaned if part)


def _vector_to_list(vector: Any) -> List[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()

    if isinstance(vector, (str, bytes, bytearray, Mapping)):
        raise ValueError("embedding vector values must be numeric")

    values = list(vector)
    if any(isinstance(value, bool) for value in values):
        raise ValueError("embedding vector values must be numeric")
    float_values = [float(value) for value in values]
    if any(not math.isfinite(value) for value in float_values):
        raise ValueError("embedding vector values must be finite numbers")
    return float_values


def _encoded_vectors_to_lists(
    encoded_vectors: Any,
    expected_count: int,
    embedding_type: str,
) -> List[List[float]]:
    if hasattr(encoded_vectors, "tolist"):
        encoded_vectors = encoded_vectors.tolist()
    if isinstance(encoded_vectors, Mapping):
        raise ValueError(f"{embedding_type} embeddings must be numeric vectors")
    try:
        vectors = [_vector_to_list(vector) for vector in encoded_vectors]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{embedding_type} embeddings must be numeric vectors") from exc

    if len(vectors) != expected_count:
        raise ValueError(
            f"{embedding_type} embedding count {len(vectors)} "
            f"does not match segment count {expected_count}"
        )

    if vectors:
        dimension_count = len(vectors[0])
        if dimension_count == 0:
            raise ValueError(f"{embedding_type} embeddings must be non-empty vectors")
        if any(len(vector) != dimension_count for vector in vectors):
            raise ValueError(
                f"{embedding_type} embeddings must have consistent dimensions"
            )

    return vectors


def _document_id(video_filename: str, segment_id: str, suffix: str) -> str:
    return f"{video_filename}{DOCUMENT_ID_SCOPE_DELIMITER}{segment_id}{suffix}"


def _delete_stale_video_documents(
    collection: VectorCollection,
    video_filename: str,
    current_ids: set[str],
) -> int:
    results = collection.get(
        where={"video_filename": video_filename},
        include=[],
    )
    ids = results.get("ids") if isinstance(results, dict) else None
    if not isinstance(ids, list):
        logger.warning(
            "Skipping stale index cleanup for %s: Chroma get returned malformed ids.",
            video_filename,
        )
        return 0

    stale_ids = []
    seen_stale_ids = set()
    for doc_id in ids:
        if not isinstance(doc_id, str):
            continue
        if doc_id in current_ids or doc_id in seen_stale_ids:
            continue
        stale_ids.append(doc_id)
        seen_stale_ids.add(doc_id)

    if not stale_ids:
        return 0

    collection.delete(ids=stale_ids)
    return len(stale_ids)


def _normalize_video_filename(video_filename: Any) -> str:
    if not isinstance(video_filename, str) or not video_filename.strip():
        raise ValueError("video_filename must be a non-empty string")
    return video_filename.strip()


def _segment_time_value(segment: Dict[str, Any], field_name: str, index: int) -> float:
    if field_name not in segment:
        raise ValueError(
            f"enriched segment at index {index} must have {field_name}"
        )

    value = segment[field_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"enriched segment at index {index} {field_name} must be a number"
        )

    time_value = float(value)
    if not math.isfinite(time_value):
        raise ValueError(
            f"enriched segment at index {index} {field_name} must be a finite number"
        )

    if time_value < 0:
        raise ValueError(
            f"enriched segment at index {index} {field_name} must be non-negative"
        )

    return time_value


def _validate_optional_string_field(
    segment: Dict[str, Any],
    index: int,
    field_name: str,
) -> None:
    if field_name in segment and not isinstance(segment[field_name], str):
        raise ValueError(
            f"enriched segment at index {index} {field_name} must be a string"
        )


def _validate_optional_string_list_field(
    segment: Dict[str, Any],
    index: int,
    field_name: str,
) -> None:
    if field_name not in segment:
        return

    value = segment[field_name]
    if not isinstance(value, list):
        raise ValueError(
            f"enriched segment at index {index} {field_name} must be a JSON array"
        )

    for item_index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(
                f"enriched segment at index {index} {field_name} item at index "
                f"{item_index} must be a string"
            )


def _normalize_required_string_field(
    segment: Dict[str, Any],
    index: int,
    field_name: str,
) -> None:
    value = segment.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"enriched segment at index {index} {field_name} "
            "must be a non-empty string"
        )
    segment[field_name] = value.strip()


def _normalize_required_string_list_field(
    segment: Dict[str, Any],
    index: int,
    field_name: str,
) -> None:
    value = segment.get(field_name)
    if not isinstance(value, list):
        raise ValueError(
            f"enriched segment at index {index} {field_name} must be a JSON array"
        )

    normalized_values = []
    for item_index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(
                f"enriched segment at index {index} {field_name} item at index "
                f"{item_index} must be a string"
            )
        item = item.strip()
        if item:
            normalized_values.append(item)

    if not normalized_values:
        raise ValueError(
            f"enriched segment at index {index} {field_name} "
            "must include at least one non-empty string"
        )
    segment[field_name] = normalized_values


def _validate_segments(segments: Any) -> List[Dict[str, Any]]:
    if not isinstance(segments, list):
        raise ValueError("enriched segments file must contain a JSON array")

    seen_segment_ids = set()
    previous_end_time: float | None = None
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"enriched segment at index {index} must be a JSON object")

        segment_id = segment.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id.strip():
            raise ValueError(f"enriched segment at index {index} must have a segment_id")
        normalized_segment_id = segment_id.strip()
        if DOCUMENT_ID_SCOPE_DELIMITER in normalized_segment_id:
            raise ValueError(
                f"enriched segment at index {index} segment_id must not contain "
                f"{DOCUMENT_ID_SCOPE_DELIMITER}"
            )
        if normalized_segment_id in seen_segment_ids:
            raise ValueError(
                f"enriched segment at index {index} has duplicate segment_id"
            )
        seen_segment_ids.add(normalized_segment_id)
        segment["segment_id"] = normalized_segment_id

        for field_name in ("full_transcript", "summary", "title"):
            _validate_optional_string_field(segment, index, field_name)
        for field_name in (
            "speakers",
            "keywords",
            "consolidated_visual_captions",
            "consolidated_actions",
            "consolidated_audio_events",
        ):
            _validate_optional_string_list_field(segment, index, field_name)

        start_time = _segment_time_value(segment, "start_time", index)
        end_time = _segment_time_value(segment, "end_time", index)
        if end_time < start_time:
            raise ValueError(
                f"enriched segment at index {index} "
                "end_time must be greater than or equal to start_time"
            )
        if previous_end_time is not None and start_time < previous_end_time:
            raise ValueError(
                f"enriched segment at index {index} overlaps previous segment"
            )
        previous_end_time = end_time
        segment["start_time"] = start_time
        segment["end_time"] = end_time
        _normalize_required_string_field(segment, index, "title")
        _normalize_required_string_field(segment, index, "summary")
        _normalize_required_string_list_field(segment, index, "keywords")

    return segments


def _prepare_metadata_for_db(segment: Dict[str, Any], video_filename: str) -> Dict[str, Any]:
    """
    Prepares a clean metadata dictionary for ChromaDB.
    ChromaDB metadata values must be strings, ints, floats, or bools.
    This metadata will be attached to BOTH the text and visual entries.
    """
    # Join lists into comma-separated strings for filtering
    speakers_str = _join_metadata_values(segment.get("speakers", []))
    keywords_str = _join_metadata_values(segment.get("keywords", []))
    actions_str = _join_metadata_values(segment.get("consolidated_actions", []))
    audio_events_str = _join_metadata_values(segment.get("consolidated_audio_events", []))

    return {
        # Core searchable metadata
        "title": str(segment.get("title", "")),
        "summary": str(segment.get("summary", "")),
        "speakers": speakers_str,
        "keywords": keywords_str,
        "actions": actions_str,
        "audio_events": audio_events_str,
        # Essential data for displaying results
        "start_time": segment["start_time"],
        "end_time": segment["end_time"],
        "video_filename": video_filename,
    }


def create_embedding_model(config: Dict[str, Any]) -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    device = config["general"]["device"]
    embedding_model_name = config["models"]["embedding"]["name"]
    return SentenceTransformer(embedding_model_name, device=device)


def create_vector_collection(config: Dict[str, Any]) -> VectorCollection:
    import chromadb

    db_config = config["database"]
    client = chromadb.HttpClient(host=db_config["host"], port=db_config["port"])
    return client.get_or_create_collection(
        name=db_config["collection_name"],
        metadata={"hnsw:space": "cosine"},
    )


def run_indexing(
    enriched_segments_path: str,
    video_filename: str,
    config: Dict[str, Any],
    embedding_model: Optional[EmbeddingModel] = None,
    collection: Optional[VectorCollection] = None,
) -> bool:
    """
    Takes final enriched segments, creates separate text and visual vector embeddings,
    and indexes them in a single ChromaDB collection with distinct IDs.
    """
    logger.info("--- Starting Step 4: Indexing Segments in Vector DB ---")
    video_filename = _normalize_video_filename(video_filename)

    # 1. Load the enriched segments data
    logger.info(f"1/4: Loading enriched segments from {enriched_segments_path}...")
    with open(enriched_segments_path, "r") as f:
        segments = _validate_segments(json.load(f))

    if not segments:
        logger.warning("No segments found in the input file. Skipping indexing.")
        return False

    # 2. Initialize the embedding model and ChromaDB client
    logger.info("2/4: Initializing models and DB client...")
    if embedding_model is None:
        embedding_model = create_embedding_model(config)
    if collection is None:
        collection = create_vector_collection(config)

    # 3. Prepare data for batch insertion
    logger.info(f"3/4: Preparing {len(segments)} segments for batch indexing...")

    # We will create two sets of entries: one for text, one for visuals.
    ids_to_add = []
    embeddings_to_add = []
    metadatas_to_add = []

    # Create all embeddings at once for GPU efficiency.
    # Text side: title (concept) + keywords (semantic anchors) + transcript
    # (or summary fallback for silent segments). Order matters because the
    # sentence-transformer truncates at ~512 tokens — putting the
    # short/concept-bearing parts first means a long transcript clips its
    # own tail rather than dropping the LLM-derived anchors entirely.
    all_text_contexts = []
    for seg in segments:
        text_parts: List[str] = [seg.get("title", "")]
        text_parts.extend(seg.get("keywords", []))
        text_parts.append(
            seg.get("full_transcript", "").strip()
            or seg.get("summary", "").strip()
        )
        all_text_contexts.append(_join_embedding_parts(text_parts))

    all_visual_contexts = []
    for seg in segments:
        visual_parts: List[str] = [seg.get("summary", "")]
        visual_parts.extend(seg.get("consolidated_visual_captions", []))
        visual_parts.extend(seg.get("consolidated_actions", []))
        visual_parts.extend(seg.get("consolidated_audio_events", []))
        all_visual_contexts.append(_join_embedding_parts(visual_parts))

    text_embeddings = _encoded_vectors_to_lists(
        embedding_model.encode(all_text_contexts, show_progress_bar=True),
        len(segments),
        "text",
    )
    visual_embeddings = _encoded_vectors_to_lists(
        embedding_model.encode(all_visual_contexts, show_progress_bar=True),
        len(segments),
        "visual",
    )

    for i, segment in enumerate(segments):
        segment_id = segment["segment_id"]
        common_metadata = _prepare_metadata_for_db(segment, video_filename)

        # --- Create the TEXT entry ---
        # We create a unique ID for the text embedding of this segment.
        ids_to_add.append(_document_id(video_filename, segment_id, "_text"))
        embeddings_to_add.append(text_embeddings[i])
        # Add a type identifier to the metadata for filtering
        text_metadata = common_metadata.copy()
        text_metadata["type"] = "text"
        metadatas_to_add.append(text_metadata)

        # --- Create the VISUAL entry ---
        # We create a unique ID for the visual embedding of this segment.
        ids_to_add.append(_document_id(video_filename, segment_id, "_visual"))
        embeddings_to_add.append(visual_embeddings[i])
        # Add a type identifier to the metadata for filtering
        visual_metadata = common_metadata.copy()
        visual_metadata["type"] = "visual"
        metadatas_to_add.append(visual_metadata)

    # 4. Add the data to ChromaDB in a single batch operation
    # This adds 2 * num_segments entries to the database.
    db_config = config["database"]
    logger.info(
        "4/4: Indexing %s documents into ChromaDB collection '%s'...",
        len(ids_to_add),
        db_config["collection_name"],
    )

    # Use 'upsert' to safely add or update entries without duplication errors.
    collection.upsert(
        ids=ids_to_add,
        embeddings=embeddings_to_add,
        metadatas=metadatas_to_add,
    )
    deleted_count = _delete_stale_video_documents(
        collection,
        video_filename,
        set(ids_to_add),
    )
    if deleted_count:
        logger.info(
            "Deleted %s stale ChromaDB documents for reindexed video %s.",
            deleted_count,
            video_filename,
        )

    logger.info("--- Indexing Step Complete! ---")
    logger.info(f"Successfully indexed text and visual data for {len(segments)} segments.")
    return True
