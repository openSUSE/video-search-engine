from typing import Any, Dict

DOCUMENT_ID_SCOPE_DELIMITER = "::"


def is_usable_video_filename_scope(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if not value or value != value.strip():
        return False
    if value in {".", ".."}:
        return False
    return "/" not in value and "\\" not in value


def is_usable_segment_id(segment_id: Any) -> bool:
    if not isinstance(segment_id, str):
        return False
    if not segment_id or segment_id != segment_id.strip():
        return False
    if DOCUMENT_ID_SCOPE_DELIMITER not in segment_id:
        return True

    video_scope, local_segment_id = segment_id.rsplit(
        DOCUMENT_ID_SCOPE_DELIMITER,
        1,
    )
    return bool(
        video_scope
        and is_usable_video_filename_scope(video_scope)
        and local_segment_id
        and local_segment_id == local_segment_id.strip()
    )


def text_metadata_by_segment_id(
    ids: Any, metadatas: Any
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(ids, list) or not isinstance(metadatas, list):
        return {}

    metadata_by_segment_id = {}
    for doc_id, metadata in zip(ids, metadatas):
        if not isinstance(doc_id, str) or not doc_id.endswith("_text"):
            continue
        if not isinstance(metadata, dict):
            continue
        segment_id = doc_id.removesuffix("_text")
        if not is_usable_segment_id(segment_id):
            continue
        if segment_id in metadata_by_segment_id:
            continue
        metadata_by_segment_id[segment_id] = metadata

    return metadata_by_segment_id
