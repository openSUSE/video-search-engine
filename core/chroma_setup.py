"""Single source of truth for Chroma collection topology.

Both the search API and the indexing pipeline previously did their own
``get_or_create_collection(metadata={"hnsw:space":"cosine"})`` calls. If a
collection already existed with a different distance (e.g. the default
``l2``), neither side raised: chromadb silently returned the existing
collection, queries used the wrong distance, and ranking subtly degraded.

This module wraps the create-or-fetch flow with an explicit check that
the existing collection's space matches what we expect, and raises if it
doesn't. Either side may now call ``get_search_collection`` and trust the
shape of what comes back.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Protocol


logger = logging.getLogger(__name__)

EXPECTED_HNSW_SPACE = "cosine"
HNSW_SPACE_METADATA_KEY = "hnsw:space"


class _Collection(Protocol):
    metadata: Dict[str, Any] | None
    name: str


def _existing_space(collection: _Collection) -> str | None:
    metadata = getattr(collection, "metadata", None) or {}
    space = metadata.get(HNSW_SPACE_METADATA_KEY)
    return space.strip() if isinstance(space, str) else None


def get_search_collection(client: Any, collection_name: str) -> Any:
    """Return the search collection, asserting the distance space.

    On a fresh Chroma the collection is created with ``hnsw:space:cosine``.
    On a populated Chroma we verify the collection's metadata matches.
    Mismatch raises rather than degrading silently — re-creating a
    collection in-place is destructive and we'd rather force the operator
    to make that call explicitly.
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={HNSW_SPACE_METADATA_KEY: EXPECTED_HNSW_SPACE},
    )
    space = _existing_space(collection)
    if space and space != EXPECTED_HNSW_SPACE:
        raise RuntimeError(
            f"Chroma collection {collection_name!r} exists with hnsw:space={space!r}; "
            f"expected {EXPECTED_HNSW_SPACE!r}. Delete and re-create it (e.g. via "
            "`chromadb` admin) before continuing — querying a cosine-encoded "
            "index against an l2 collection silently degrades ranking."
        )
    if space is None:
        logger.info(
            "Chroma collection %r has no hnsw:space metadata; assuming "
            "freshly created with %s.",
            collection_name,
            EXPECTED_HNSW_SPACE,
        )
    return collection
