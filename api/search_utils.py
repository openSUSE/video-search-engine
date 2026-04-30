from typing import Any, Dict


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
        if not segment_id.strip():
            continue
        if segment_id != segment_id.strip():
            continue
        metadata_by_segment_id[segment_id] = metadata

    return metadata_by_segment_id
