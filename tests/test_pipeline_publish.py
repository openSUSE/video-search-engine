"""Unit tests for the Streamlit-side publish helper.

The publisher itself is exercised in tests/test_ingestion_jobs.py against
a fake pika; here we just verify the host→worker path translation and
the error-to-result conversion that the Streamlit page depends on.
"""

from __future__ import annotations

import sys
import types

import pytest

from app.ui.pipeline_publish import (
    PublishResult,
    WORKER_VIDEO_DIR,
    _worker_video_path,
    publish_ingestion_job,
)


def test_worker_video_path_translates_inside_data_videos(tmp_path):
    host_dir = tmp_path / "videos"
    host_dir.mkdir()
    host_video = host_dir / "demo.mp4"
    host_video.touch()
    assert _worker_video_path(str(host_video), str(host_dir)) == (
        f"{WORKER_VIDEO_DIR}/demo.mp4"
    )


def test_worker_video_path_passes_through_when_outside_videos_dir(tmp_path):
    host_dir = tmp_path / "videos"
    host_dir.mkdir()
    other = tmp_path / "elsewhere" / "clip.mov"
    other.parent.mkdir()
    other.touch()
    assert _worker_video_path(str(other), str(host_dir)) == str(other)


def _install_fake_pika(monkeypatch, *, raise_on_publish: Exception | None = None):
    """Provide a minimal fake pika so publish_ingestion_job runs without
    a live RabbitMQ. Mirrors what tests/test_ingestion_jobs.py does."""

    fake_pika = types.ModuleType("pika")
    fake_exceptions = types.ModuleType("pika.exceptions")

    class _ChannelClosedByBroker(Exception):
        def __init__(self, reply_code, reply_text):
            super().__init__(reply_text)
            self.reply_code = reply_code
            self.reply_text = reply_text

    fake_exceptions.ChannelClosedByBroker = _ChannelClosedByBroker

    class _BasicProperties:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Spec:
        PERSISTENT_DELIVERY_MODE = 2

    fake_pika.exceptions = fake_exceptions
    fake_pika.BasicProperties = _BasicProperties
    fake_pika.spec = _Spec

    class FakeChannel:
        def exchange_declare(self, *, exchange, exchange_type, durable):
            pass

        def queue_bind(self, *, queue, exchange, routing_key):
            pass

        def queue_declare(self, *, queue, durable, arguments=None):
            pass

        def basic_publish(self, *, exchange, routing_key, body, properties):
            if raise_on_publish is not None:
                raise raise_on_publish

    class FakeConnection:
        is_open = True

        def channel(self):
            return FakeChannel()

        def close(self):
            pass

    class FakeURLParameters:
        def __init__(self, url):
            self.url = url

    fake_pika.URLParameters = FakeURLParameters
    fake_pika.BlockingConnection = lambda params: FakeConnection()

    monkeypatch.setitem(sys.modules, "pika", fake_pika)
    monkeypatch.setitem(sys.modules, "pika.exceptions", fake_exceptions)


def test_publish_ingestion_job_returns_ok_result(monkeypatch, tmp_path):
    _install_fake_pika(monkeypatch)
    monkeypatch.setenv("RABBITMQ_URL", "amqp://user:pw@localhost:5672/%2F")

    result = publish_ingestion_job(
        host_video_path=str(tmp_path / "videos" / "x.mp4"),
        host_video_dir=str(tmp_path / "videos"),
        title="X",
    )
    assert isinstance(result, PublishResult)
    assert result.ok is True
    assert "x.mp4" in result.job_video_path
    assert result.queue_name == "video.ingestion"


def test_publish_ingestion_job_handles_invalid_url(monkeypatch, tmp_path):
    _install_fake_pika(monkeypatch)
    monkeypatch.setenv("RABBITMQ_URL", "not-a-valid-url")

    result = publish_ingestion_job(
        host_video_path=str(tmp_path / "videos" / "x.mp4"),
        host_video_dir=str(tmp_path / "videos"),
    )
    assert result.ok is False
    assert "RabbitMQ" in result.detail or "Invalid" in result.detail


def test_publish_ingestion_job_handles_publish_exception(monkeypatch, tmp_path):
    _install_fake_pika(monkeypatch, raise_on_publish=RuntimeError("broker down"))
    monkeypatch.setenv("RABBITMQ_URL", "amqp://user:pw@localhost:5672/%2F")

    result = publish_ingestion_job(
        host_video_path=str(tmp_path / "videos" / "x.mp4"),
        host_video_dir=str(tmp_path / "videos"),
    )
    assert result.ok is False
    assert "broker down" in result.detail
