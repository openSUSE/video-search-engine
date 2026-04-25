# ingestion_pipeline/steps/step_04_indexing.py

import json
import logging
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    def encode(self, sentences: List[str], **kwargs: Any) -> Any:
        ...


class VectorCollection(Protocol):
    def upsert(self, **kwargs: Any) -> Any:
        ...


def _join_metadata_values(values: Any) -> str:
    if not values:
        return ""
    if isinstance(values, str):
        return values
    return ",".join(str(value) for value in values)


def _vector_to_list(vector: Any) -> List[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return list(vector)


def _encoded_vectors_to_lists(
    encoded_vectors: Any,
    expected_count: int,
    embedding_type: str,
) -> List[List[float]]:
    if hasattr(encoded_vectors, "tolist"):
        encoded_vectors = encoded_vectors.tolist()
    try:
        vectors = [_vector_to_list(vector) for vector in encoded_vectors]
    except TypeError as exc:
        raise ValueError(f"{embedding_type} embeddings must be vectors") from exc

    if len(vectors) != expected_count:
        raise ValueError(
            f"{embedding_type} embedding count {len(vectors)} "
            f"does not match segment count {expected_count}"
        )

    return vectors


def _document_id(video_filename: str, segment_id: str, suffix: str) -> str:
    return f"{video_filename}::{segment_id}{suffix}"


def _normalize_video_filename(video_filename: Any) -> str:
    if not isinstance(video_filename, str) or not video_filename.strip():
        raise ValueError("video_filename must be a non-empty string")
    return video_filename.strip()


def _segment_time_value(segment: Dict[str, Any], field_name: str, index: int) -> float:
    value = segment.get(field_name, 0.0)
    if isinstance(value, bool):
        raise ValueError(
            f"enriched segment at index {index} {field_name} must be a number"
        )

    try:
        time_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"enriched segment at index {index} {field_name} must be a number"
        ) from exc

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


def _validate_segments(segments: Any) -> List[Dict[str, Any]]:
    if not isinstance(segments, list):
        raise ValueError("enriched segments file must contain a JSON array")

    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"enriched segment at index {index} must be a JSON object")

        segment_id = segment.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id.strip():
            raise ValueError(f"enriched segment at index {index} must have a segment_id")

        for field_name in ("full_transcript", "summary", "title"):
            _validate_optional_string_field(segment, index, field_name)
        for field_name in (
            "speakers",
            "keywords",
            "consolidated_visual_captions",
            "consolidated_actions",
        ):
            _validate_optional_string_list_field(segment, index, field_name)

        start_time = _segment_time_value(segment, "start_time", index)
        end_time = _segment_time_value(segment, "end_time", index)
        if "start_time" in segment and "end_time" in segment and end_time < start_time:
            raise ValueError(
                f"enriched segment at index {index} "
                "end_time must be greater than or equal to start_time"
            )

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

    return {
        # Core searchable metadata
        "title": str(segment.get("title", "")),
        "summary": str(segment.get("summary", "")),
        "speakers": speakers_str,
        "keywords": keywords_str,
        "actions": actions_str,
        # Essential data for displaying results
        "start_time": float(segment.get("start_time", 0.0)),
        "end_time": float(segment.get("end_time", 0.0)),
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
) -> None:
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
        return

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

    # Create all embeddings at once for GPU efficiency
    all_transcripts = [
        seg.get("full_transcript", "") or seg.get("summary", "")
        for seg in segments
    ]

    all_visual_contexts = []
    for seg in segments:
        visual_parts = [seg.get("summary", "")]
        visual_parts.extend(seg.get("consolidated_visual_captions", []))
        visual_parts.extend(seg.get("consolidated_actions", []))
        # Join non-empty parts with a period for semantic separation
        all_visual_contexts.append(". ".join(filter(None, visual_parts)))

    text_embeddings = _encoded_vectors_to_lists(
        embedding_model.encode(all_transcripts, show_progress_bar=True),
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

    logger.info("--- Indexing Step Complete! ---")
    logger.info(f"Successfully indexed text and visual data for {len(segments)} segments.")
