# ingestion_pipeline/steps/step_04_indexing.py

import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Protocol

from ingestion_pipeline.jobs import normalize_required_string

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


def _content_hash(text: str, model_name: str) -> str:
    """SHA-256 hash that pins (model, content) so re-indexing skips stable rows."""
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _existing_context_hashes(
    collection: VectorCollection, doc_ids: List[str]
) -> Dict[str, str]:
    """Return a map of doc_id -> stored context_hash (when present).

    Missing IDs and IDs without a recorded hash are simply absent from the
    map. The caller treats absence as "needs (re)encode".
    """
    if not doc_ids:
        return {}
    try:
        result = collection.get(ids=doc_ids, include=["metadatas"])
    except Exception as exc:
        logger.warning(
            "Could not fetch existing context hashes (%s); proceeding as if "
            "no rows exist.",
            exc,
        )
        return {}
    if not isinstance(result, dict):
        return {}
    ids = result.get("ids") or []
    metas = result.get("metadatas") or []
    out: Dict[str, str] = {}
    for doc_id, meta in zip(ids, metas):
        if not isinstance(doc_id, str) or not isinstance(meta, dict):
            continue
        stored_hash = meta.get("context_hash")
        if isinstance(stored_hash, str) and stored_hash:
            out[doc_id] = stored_hash
    return out


def _encode_normalized(embedding_model: EmbeddingModel, sentences: List[str]) -> Any:
    """Encode with normalize_embeddings=True when the model supports it.

    Falls back through progressively narrower signatures so test stubs that
    only accept (texts, show_progress_bar) — or just (texts) — still work.
    """
    for kwargs in (
        {"show_progress_bar": True, "normalize_embeddings": True},
        {"show_progress_bar": True},
        {"normalize_embeddings": True},
        {},
    ):
        try:
            return embedding_model.encode(sentences, **kwargs)
        except TypeError:
            continue
    # Should not reach here unless the stub's signature is broken.
    return embedding_model.encode(sentences)


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
        if DOCUMENT_ID_SCOPE_DELIMITER in doc_id:
            document_video_filename, _segment_id = doc_id.rsplit(
                DOCUMENT_ID_SCOPE_DELIMITER,
                1,
            )
            if document_video_filename != video_filename:
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
    invalid_message = "video_filename must be a non-empty filename"
    if not isinstance(video_filename, str):
        raise ValueError(invalid_message)
    video_filename = video_filename.strip()
    if (
        not video_filename
        or video_filename in {".", ".."}
        or os.path.basename(video_filename) != video_filename
        or "\\" in video_filename
    ):
        raise ValueError(invalid_message)
    return video_filename


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


def _tokenized_metadata_value(values: Any) -> str:
    """Lowercase, space-padded list join for ChromaDB ``$contains`` queries.

    ChromaDB metadata is scalar (string / int / float / bool); list fields
    have to be flattened. The previous comma-join meant ``where: {"speakers":
    "Sintel"}`` only matched the *whole joined string* — useless when
    a segment has multiple speakers. By emitting tokens separated by ``" | "``
    with leading/trailing pipes, callers can use ``$contains`` to match a
    single token unambiguously, e.g.
    ``where: {"speakers_tokens": {"$contains": "|sintel|"}}``.
    """
    if not values:
        return ""
    if isinstance(values, str):
        token = values.strip().lower()
        return f"|{token}|" if token else ""
    cleaned = [str(value).strip().lower() for value in values]
    cleaned = [value for value in cleaned if value]
    if not cleaned:
        return ""
    return "|" + "|".join(cleaned) + "|"


def _prepare_metadata_for_db(segment: Dict[str, Any], video_filename: str) -> Dict[str, Any]:
    """
    Prepares a clean metadata dictionary for ChromaDB.
    ChromaDB metadata values must be strings, ints, floats, or bools.
    This metadata will be attached to BOTH the text and visual entries.

    Two forms are emitted for list fields:
    - the comma-joined human-readable string (e.g. ``speakers``) for display
    - a pipe-delimited lowercase token string (e.g. ``speakers_tokens``) for
      precise ``$contains`` filtering. The token form is the only metadata
      shape that can answer "list me segments where speaker X appears".
    """
    speakers_str = _join_metadata_values(segment.get("speakers", []))
    keywords_str = _join_metadata_values(segment.get("keywords", []))
    actions_str = _join_metadata_values(segment.get("consolidated_actions", []))
    audio_events_str = _join_metadata_values(segment.get("consolidated_audio_events", []))

    return {
        # Core searchable metadata
        "title": str(segment.get("title", "")),
        "summary": str(segment.get("summary", "")),
        "speakers": speakers_str,
        "speakers_tokens": _tokenized_metadata_value(segment.get("speakers", [])),
        "keywords": keywords_str,
        "keywords_tokens": _tokenized_metadata_value(segment.get("keywords", [])),
        "actions": actions_str,
        "actions_tokens": _tokenized_metadata_value(segment.get("consolidated_actions", [])),
        "audio_events": audio_events_str,
        "audio_events_tokens": _tokenized_metadata_value(segment.get("consolidated_audio_events", [])),
        # Essential data for displaying results
        "start_time": segment["start_time"],
        "end_time": segment["end_time"],
        "duration_sec": float(segment["end_time"]) - float(segment["start_time"]),
        "video_filename": video_filename,
    }


def create_embedding_model(config: Dict[str, Any]) -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    device = config["general"]["device"]
    embedding_model_name = config["models"]["embedding"]["name"]
    return SentenceTransformer(embedding_model_name, device=device)


def create_vector_collection(config: Dict[str, Any]) -> VectorCollection:
    import chromadb

    from core.chroma_setup import get_search_collection

    db_config = config["database"]
    client = chromadb.HttpClient(host=db_config["host"], port=db_config["port"])
    return get_search_collection(client, db_config["collection_name"])


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
    try:
        enriched_segments_path = normalize_required_string(
            enriched_segments_path,
            "enriched_segments_path",
        )
    except ValueError as exc:
        logger.error("Invalid indexing input: %s", exc)
        return False

    video_filename = _normalize_video_filename(video_filename)

    # 1. Load the enriched segments data
    logger.info(f"1/4: Loading enriched segments from {enriched_segments_path}...")
    try:
        with open(enriched_segments_path, "r") as f:
            loaded_segments = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error(
            "Could not read or parse enriched segments at %s: %s",
            enriched_segments_path,
            exc,
        )
        return False

    segments = _validate_segments(loaded_segments)

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

    # Hash the contexts under the embedding model name. If every doc id
    # already exists in the collection with a matching hash, the work is
    # idempotent and we skip the encode + upsert entirely. This dominates
    # rerun cost on a stable catalog: re-running indexing after fixing a
    # typo in one segment shouldn't force a re-encode of every other one.
    embedding_model_name = (
        config.get("models", {}).get("embedding", {}).get("name") or "unknown"
    )
    text_hashes = [_content_hash(ctx, embedding_model_name) for ctx in all_text_contexts]
    visual_hashes = [_content_hash(ctx, embedding_model_name) for ctx in all_visual_contexts]

    text_doc_ids = [
        _document_id(video_filename, seg["segment_id"], "_text")
        for seg in segments
    ]
    visual_doc_ids = [
        _document_id(video_filename, seg["segment_id"], "_visual")
        for seg in segments
    ]
    existing_hashes = _existing_context_hashes(
        collection,
        text_doc_ids + visual_doc_ids,
    )
    all_text_unchanged = all(
        existing_hashes.get(text_doc_ids[i]) == text_hashes[i]
        for i in range(len(segments))
    )
    all_visual_unchanged = all(
        existing_hashes.get(visual_doc_ids[i]) == visual_hashes[i]
        for i in range(len(segments))
    )
    if all_text_unchanged and all_visual_unchanged:
        logger.info(
            "All %d segments already indexed with matching context hashes; "
            "skipping re-encode and upsert.",
            len(segments),
        )
        return True

    # normalize_embeddings=True keeps both index and query vectors on the
    # unit sphere; the collection uses hnsw:space:cosine and the search
    # service normalizes its query vector, so consistent normalization here
    # makes recall stable across reindex runs and removes a small amount of
    # internal recomputation Chroma does on un-normalized inputs.
    text_embeddings = _encoded_vectors_to_lists(
        _encode_normalized(embedding_model, all_text_contexts),
        len(segments),
        "text",
    )
    visual_embeddings = _encoded_vectors_to_lists(
        _encode_normalized(embedding_model, all_visual_contexts),
        len(segments),
        "visual",
    )

    for i, segment in enumerate(segments):
        segment_id = segment["segment_id"]
        common_metadata = _prepare_metadata_for_db(segment, video_filename)

        # --- Create the TEXT entry ---
        # We create a unique ID for the text embedding of this segment.
        ids_to_add.append(text_doc_ids[i])
        embeddings_to_add.append(text_embeddings[i])
        # Add a type identifier to the metadata for filtering
        text_metadata = common_metadata.copy()
        text_metadata["type"] = "text"
        text_metadata["context_hash"] = text_hashes[i]
        metadatas_to_add.append(text_metadata)

        # --- Create the VISUAL entry ---
        # We create a unique ID for the visual embedding of this segment.
        ids_to_add.append(visual_doc_ids[i])
        embeddings_to_add.append(visual_embeddings[i])
        # Add a type identifier to the metadata for filtering
        visual_metadata = common_metadata.copy()
        visual_metadata["type"] = "visual"
        visual_metadata["context_hash"] = visual_hashes[i]
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
