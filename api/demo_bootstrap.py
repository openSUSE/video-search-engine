"""Auto-populate Chroma with a demo video on first API boot.

Goal: someone clones the repo, runs ``make compose-up``, and immediately
sees working search results — no HF token, no Gemini key, no worker, no
ingestion run required. The repo ships a tiny pre-processed Sintel
trailer under ``app/demo/`` so we can bypass every step of the pipeline
that needs an external dependency.

Behavior:

* If ``DEMO_BOOTSTRAP`` is "0" / "false" / unset-and-explicit-off, do
  nothing. Default is on so the docker-compose defaults give a working
  demo out of the box.
* If the Chroma collection already has rows for the demo video,
  bootstrap is skipped — re-indexing on every API restart would be
  wasteful and would also hide a real production catalog if a sysadmin
  forgot to flip the flag off.
* Otherwise, run ``run_indexing()`` against the bundled enriched
  segments JSON. The resulting docs are exactly what the worker would
  have produced for that video, so the search service can't tell the
  difference.
* The demo video file itself is symlinked into ``VIDEO_DATA_PATH`` if
  the search UI's media dir is writable, so ``st.video()`` can play it.
  When the dir is read-only (the api container mounts it ro-by-default)
  we skip the symlink — the search results still render, just without
  an in-page player.

This module is import-safe — it doesn't load any heavy ML deps until
``bootstrap()`` is actually called.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEMO_VIDEO_FILENAME = "sintel_trailer"
DEMO_VIDEO_BASENAME = f"{DEMO_VIDEO_FILENAME}.mp4"
DEMO_ENRICHED_BASENAME = f"{DEMO_VIDEO_FILENAME}.enriched.json"


def _demo_root() -> str:
    """Repo-root-relative path to ``app/demo/``."""
    return os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        "app",
        "demo",
    )


def _bootstrap_enabled() -> bool:
    raw = (os.environ.get("DEMO_BOOTSTRAP") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _publish_demo_video_to_data_dir(video_data_path: str) -> Optional[str]:
    """Copy the demo MP4 into VIDEO_DATA_PATH so the UI can play it.

    Returns the published path on success, or None when the source is
    missing or the dest is not writable. Read-only mounts are not an
    error condition — we just skip the player and let search results
    surface as text.
    """
    src = os.path.join(_demo_root(), DEMO_VIDEO_BASENAME)
    if not os.path.exists(src):
        logger.warning("Demo video missing at %s; skipping copy.", src)
        return None
    if not video_data_path:
        return None
    try:
        os.makedirs(video_data_path, exist_ok=True)
    except OSError as exc:
        logger.info(
            "Demo video target %s not writable (%s); skipping copy.",
            video_data_path,
            exc,
        )
        return None
    dest = os.path.join(video_data_path, DEMO_VIDEO_BASENAME)
    if os.path.exists(dest):
        return dest
    try:
        shutil.copyfile(src, dest)
    except OSError as exc:
        logger.info(
            "Could not publish demo video to %s (%s); the search UI player "
            "will be empty but search results still work.",
            dest,
            exc,
        )
        return None
    logger.info("Demo video published to %s.", dest)
    return dest


def _collection_has_video(collection: Any, video_filename: str) -> bool:
    """Returns True when the Chroma collection has any rows for the video.

    Errors are treated as "unknown -> skip bootstrap" so we never
    accidentally overwrite a real catalog because of a transient probe
    failure.
    """
    try:
        result = collection.get(
            where={"video_filename": video_filename},
            include=[],
            limit=1,
        )
    except Exception as exc:
        logger.warning(
            "Demo bootstrap probe failed (%s); assuming collection is "
            "populated to avoid clobbering real data.",
            exc,
        )
        return True
    if not isinstance(result, dict):
        return True
    ids = result.get("ids")
    return bool(ids)


def bootstrap(config: Dict[str, Any]) -> None:
    """Idempotent demo bootstrap. Safe to call from FastAPI lifespan."""
    if not _bootstrap_enabled():
        logger.info("DEMO_BOOTSTRAP disabled; skipping demo population.")
        return

    enriched_path = os.path.join(_demo_root(), DEMO_ENRICHED_BASENAME)
    if not os.path.exists(enriched_path):
        logger.info(
            "No demo enriched segments at %s; skipping demo bootstrap.",
            enriched_path,
        )
        return

    # Build the search collection through the same authoritative helper
    # the rest of the API uses, so the hnsw:space mismatch check fires
    # here too.
    try:
        import chromadb

        from core.chroma_setup import get_search_collection
    except ImportError as exc:
        logger.warning(
            "Demo bootstrap unavailable (chromadb import failed: %s).", exc
        )
        return

    db_config = config["database"]
    try:
        client = chromadb.HttpClient(
            host=db_config["host"], port=db_config["port"]
        )
        client.heartbeat()
        collection = get_search_collection(client, db_config["collection_name"])
    except Exception as exc:
        # Bootstrap is best-effort — the API still serves /healthz etc.
        # even if Chroma is briefly unreachable.
        logger.warning("Demo bootstrap could not reach Chroma (%s).", exc)
        return

    if _collection_has_video(collection, DEMO_VIDEO_FILENAME):
        logger.info(
            "Collection already contains rows for %r; skipping demo bootstrap.",
            DEMO_VIDEO_FILENAME,
        )
        # Still publish the demo MP4 to VIDEO_DATA_PATH so the UI player
        # works after a fresh container restart.
        video_dir = (os.environ.get("VIDEO_DATA_PATH") or "").strip()
        if video_dir:
            _publish_demo_video_to_data_dir(video_dir)
        return

    logger.info("Bootstrapping demo data from %s", enriched_path)

    try:
        from ingestion_pipeline.steps.step_04_indexing import (
            create_embedding_model,
            run_indexing,
        )
    except ImportError as exc:
        logger.warning("Demo bootstrap skipped — indexing module unavailable: %s", exc)
        return

    try:
        embedding_model = create_embedding_model(config)
    except Exception as exc:
        logger.warning(
            "Demo bootstrap skipped — could not load embedding model: %s", exc
        )
        return

    ok = run_indexing(
        enriched_segments_path=enriched_path,
        video_filename=DEMO_VIDEO_FILENAME,
        config=config,
        embedding_model=embedding_model,
        collection=collection,
    )
    if ok:
        logger.info("Demo bootstrap indexed %r successfully.", DEMO_VIDEO_FILENAME)
    else:
        logger.warning("Demo bootstrap indexing reported failure for %r.", DEMO_VIDEO_FILENAME)

    video_dir = (os.environ.get("VIDEO_DATA_PATH") or "").strip()
    if video_dir:
        _publish_demo_video_to_data_dir(video_dir)
