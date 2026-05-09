"""Streamlit-side ingestion-job publisher.

Wraps ``ingestion_pipeline.jobs.publish_ingestion_job`` so the Streamlit
Submit page can hand a clean error path back to the user instead of a
pika stack trace. Also resolves the worker-internal video path for
videos under ``data/videos/`` — the worker container mounts that
directory at ``/data/videos`` regardless of the host path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


WORKER_VIDEO_DIR = "/data/videos"


@dataclass(frozen=True)
class PublishResult:
    ok: bool
    queue_name: str
    detail: str
    job_video_path: str = ""


def _worker_video_path(host_video_path: str, host_video_dir: str) -> str:
    """Translate a host path under ``host_video_dir`` to the worker mount.

    Used so the user can pick e.g. ``data/videos/sintel_trailer.mp4`` in the
    UI and the worker container sees ``/data/videos/sintel_trailer.mp4``
    when it deserialises the job. Falls back to the original path when it
    isn't under ``host_video_dir`` — that path will simply have to exist
    inside the worker container by some other means.
    """
    abs_host_dir = os.path.abspath(host_video_dir)
    abs_host_video = os.path.abspath(host_video_path)
    if abs_host_video.startswith(abs_host_dir + os.sep):
        relative = abs_host_video[len(abs_host_dir) + 1 :]
        return f"{WORKER_VIDEO_DIR}/{relative}"
    return host_video_path


def publish_ingestion_job(
    host_video_path: str,
    host_video_dir: str,
    *,
    title: Optional[str] = None,
    year: Optional[int] = None,
    rabbitmq_url: Optional[str] = None,
    queue_name: Optional[str] = None,
) -> PublishResult:
    """Publish a job and return a UI-friendly result object.

    Errors are not raised — the Streamlit page would otherwise need to
    catch every pika/jobs exception itself. We return a ``PublishResult``
    with ``ok=False`` and a one-line detail the page can ``st.error()``
    directly.
    """
    try:
        from ingestion_pipeline.jobs import (
            DEFAULT_QUEUE,
            IngestionJob,
            publish_ingestion_job as _publish,
            resolve_ingestion_queue,
            resolve_rabbitmq_url,
        )
    except ImportError as exc:
        return PublishResult(
            ok=False,
            queue_name="",
            detail=f"ingestion_pipeline import failed: {exc}",
        )

    try:
        rabbitmq_url = resolve_rabbitmq_url(rabbitmq_url)
        queue_name = resolve_ingestion_queue(queue_name)
    except ValueError as exc:
        return PublishResult(
            ok=False,
            queue_name=queue_name or DEFAULT_QUEUE,
            detail=f"Invalid RabbitMQ config: {exc}",
        )

    worker_path = _worker_video_path(host_video_path, host_video_dir)
    job = IngestionJob(video_path=worker_path, title=title, year=year)

    try:
        _publish(job, rabbitmq_url=rabbitmq_url, queue_name=queue_name)
    except Exception as exc:
        return PublishResult(
            ok=False,
            queue_name=queue_name,
            detail=f"Publish failed: {exc}",
            job_video_path=worker_path,
        )

    return PublishResult(
        ok=True,
        queue_name=queue_name,
        detail=f"Published to queue {queue_name!r}",
        job_video_path=worker_path,
    )
