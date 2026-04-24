from typing import Any, Dict, List


def text_metadata_by_segment_id(
    ids: List[str], metadatas: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    return {
        doc_id.removesuffix("_text"): metadata
        for doc_id, metadata in zip(ids, metadatas)
        if doc_id.endswith("_text")
    }
