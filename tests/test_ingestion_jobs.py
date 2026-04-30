import json

import pytest

import ingestion_pipeline.jobs as jobs
from ingestion_pipeline.jobs import (
    IngestionJob,
    decode_job_message,
    encode_job_message,
    resolve_ingestion_queue,
    resolve_rabbitmq_url,
)


def test_ingestion_job_message_round_trip_omits_empty_fields():
    job = IngestionJob(video_path="/data/videos/demo.mp4", output_dir=None, title=None, year=2024)

    message = json.loads(encode_job_message(job))
    decoded = decode_job_message(json.dumps(message).encode("utf-8"))

    assert message == {"video_path": "/data/videos/demo.mp4", "year": 2024}
    assert decoded == job


def test_ingestion_job_normalizes_optional_string_fields():
    job = IngestionJob(
        video_path="/data/videos/demo.mp4",
        output_dir="  /data/processed  ",
        title="  Demo Movie  ",
    )

    assert job.output_dir == "/data/processed"
    assert job.title == "Demo Movie"
    assert json.loads(encode_job_message(job)) == {
        "video_path": "/data/videos/demo.mp4",
        "output_dir": "/data/processed",
        "title": "Demo Movie",
    }
    assert job.to_pipeline_kwargs("fallback") == {
        "video_path": "/data/videos/demo.mp4",
        "output_dir": "/data/processed",
        "title": "Demo Movie",
        "year": None,
    }


def test_ingestion_job_normalizes_video_path():
    job = IngestionJob(video_path="  /data/videos/demo.mp4  ")

    assert job.video_path == "/data/videos/demo.mp4"
    assert json.loads(encode_job_message(job)) == {
        "video_path": "/data/videos/demo.mp4",
    }
    assert job.to_pipeline_kwargs("fallback")["video_path"] == "/data/videos/demo.mp4"


def test_ingestion_job_omits_blank_optional_string_fields():
    job = IngestionJob(
        video_path="/data/videos/demo.mp4",
        output_dir="   ",
        title="\t",
    )

    assert job.output_dir is None
    assert job.title is None
    assert json.loads(encode_job_message(job)) == {
        "video_path": "/data/videos/demo.mp4",
    }
    assert job.to_pipeline_kwargs("fallback") == {
        "video_path": "/data/videos/demo.mp4",
        "output_dir": "fallback",
        "title": None,
        "year": None,
    }


def test_ingestion_job_rejects_missing_video_path():
    with pytest.raises(ValueError, match="video_path"):
        IngestionJob(video_path="")


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"video_path": 123}, "video_path"),
        ({"video_path": "  "}, "video_path"),
        ({"video_path": "/data/videos/demo.mp4", "output_dir": 42}, "output_dir"),
        ({"video_path": "/data/videos/demo.mp4", "title": 42}, "title"),
        ({"video_path": "/data/videos/demo.mp4", "year": "2024"}, "year"),
        ({"video_path": "/data/videos/demo.mp4", "year": True}, "year"),
        ({"video_path": "/data/videos/demo.mp4", "year": 0}, "year"),
        ({"video_path": "/data/videos/demo.mp4", "year": -1}, "year"),
    ],
)
def test_decode_job_message_rejects_invalid_field_types(payload, error):
    with pytest.raises(ValueError, match=error):
        decode_job_message(json.dumps(payload).encode("utf-8"))


@pytest.mark.parametrize("payload", [{}, {"output_dir": "/data/processed"}])
def test_decode_job_message_rejects_missing_required_video_path(payload):
    with pytest.raises(ValueError, match="video_path"):
        decode_job_message(json.dumps(payload).encode("utf-8"))


def test_ingestion_job_builds_pipeline_kwargs_with_default_output_dir():
    job = IngestionJob(video_path="/data/videos/demo.mp4", title="Demo", year=2024)

    assert job.to_pipeline_kwargs("data/processed") == {
        "video_path": "/data/videos/demo.mp4",
        "output_dir": "data/processed",
        "title": "Demo",
        "year": 2024,
    }


def test_resolve_rabbitmq_url_prefers_explicit_argument(monkeypatch):
    monkeypatch.setenv("RABBITMQ_URL", "amqp://env")

    assert resolve_rabbitmq_url("amqp://explicit") == "amqp://explicit"


def test_resolve_rabbitmq_url_uses_environment(monkeypatch):
    monkeypatch.setenv("RABBITMQ_URL", "amqp://env")

    assert resolve_rabbitmq_url() == "amqp://env"


def test_resolve_rabbitmq_url_rejects_missing_value(monkeypatch):
    monkeypatch.delenv("RABBITMQ_URL", raising=False)

    with pytest.raises(ValueError, match="RabbitMQ URL"):
        resolve_rabbitmq_url()


@pytest.mark.parametrize("rabbitmq_url", ["", "   ", "\t"])
def test_resolve_rabbitmq_url_rejects_blank_explicit_values(monkeypatch, rabbitmq_url):
    monkeypatch.setenv("RABBITMQ_URL", "amqp://env")

    with pytest.raises(ValueError, match="RabbitMQ URL"):
        resolve_rabbitmq_url(rabbitmq_url)


def test_resolve_ingestion_queue_prefers_explicit_argument(monkeypatch):
    monkeypatch.setenv("INGESTION_QUEUE", "env.queue")

    assert resolve_ingestion_queue("  explicit.queue  ") == "explicit.queue"


def test_resolve_ingestion_queue_uses_environment(monkeypatch):
    monkeypatch.setenv("INGESTION_QUEUE", "  env.queue  ")

    assert resolve_ingestion_queue() == "env.queue"


def test_resolve_ingestion_queue_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("INGESTION_QUEUE", raising=False)

    assert resolve_ingestion_queue() == "video.ingestion"


@pytest.mark.parametrize("queue_name", ["", "   ", "\t"])
def test_resolve_ingestion_queue_rejects_blank_values(monkeypatch, queue_name):
    monkeypatch.delenv("INGESTION_QUEUE", raising=False)

    with pytest.raises(ValueError, match="queue"):
        resolve_ingestion_queue(queue_name)


def test_resolve_ingestion_queue_rejects_blank_environment(monkeypatch):
    monkeypatch.setenv("INGESTION_QUEUE", "  ")

    with pytest.raises(ValueError, match="queue"):
        resolve_ingestion_queue()


def test_publish_job_rejects_blank_queue_before_opening_channel(monkeypatch):
    def fail_open_channel(rabbitmq_url, queue_name):
        raise AssertionError("blank queue should be rejected before opening a channel")

    monkeypatch.setattr(jobs, "_open_channel", fail_open_channel)

    with pytest.raises(ValueError, match="queue"):
        jobs.publish_ingestion_job(
            IngestionJob(video_path="/data/videos/demo.mp4"),
            rabbitmq_url="amqp://broker",
            queue_name=" ",
        )


def test_publish_job_rejects_blank_rabbitmq_url_before_opening_channel(monkeypatch):
    def fail_open_channel(rabbitmq_url, queue_name):
        raise AssertionError("blank broker URL should be rejected before opening a channel")

    monkeypatch.setattr(jobs, "_open_channel", fail_open_channel)

    with pytest.raises(ValueError, match="RabbitMQ URL"):
        jobs.publish_ingestion_job(
            IngestionJob(video_path="/data/videos/demo.mp4"),
            rabbitmq_url=" ",
            queue_name="video.ingestion",
        )


def test_publish_job_uses_environment_queue_when_queue_omitted(monkeypatch):
    monkeypatch.setenv("INGESTION_QUEUE", "  env.queue  ")
    calls = {}

    class FakeConnection:
        is_open = True

        def close(self):
            calls["closed"] = True

    class FakeChannel:
        def basic_publish(self, *, exchange, routing_key, body, properties):
            calls["publish"] = {
                "exchange": exchange,
                "routing_key": routing_key,
                "body": body,
                "properties": properties,
            }

    def fake_open_channel(rabbitmq_url, queue_name):
        calls["open"] = {"rabbitmq_url": rabbitmq_url, "queue_name": queue_name}
        return FakeConnection(), FakeChannel()

    monkeypatch.setattr(jobs, "_open_channel", fake_open_channel)

    jobs.publish_ingestion_job(
        IngestionJob(video_path="/data/videos/demo.mp4"),
        rabbitmq_url="amqp://broker",
    )

    assert calls["open"] == {
        "rabbitmq_url": "amqp://broker",
        "queue_name": "env.queue",
    }
    assert calls["publish"]["routing_key"] == "env.queue"
    assert calls["closed"] is True


def test_consume_jobs_rejects_blank_rabbitmq_url_before_opening_channel(monkeypatch):
    def fail_open_channel(rabbitmq_url, queue_name):
        raise AssertionError("blank broker URL should be rejected before opening a channel")

    monkeypatch.setattr(jobs, "_open_channel", fail_open_channel)

    with pytest.raises(ValueError, match="RabbitMQ URL"):
        jobs.consume_ingestion_jobs(
            lambda _job: True,
            rabbitmq_url=" ",
            queue_name="video.ingestion",
        )


def test_publish_job_skips_close_when_connection_already_closed(monkeypatch):
    calls = {}

    class FakeConnection:
        is_open = False

        def close(self):
            calls["closed"] = True

    class FakeChannel:
        def basic_publish(self, *, exchange, routing_key, body, properties):
            calls["publish"] = True

    def fake_open_channel(rabbitmq_url, queue_name):
        return FakeConnection(), FakeChannel()

    monkeypatch.setattr(jobs, "_open_channel", fake_open_channel)

    jobs.publish_ingestion_job(
        IngestionJob(video_path="/data/videos/demo.mp4"),
        rabbitmq_url="amqp://broker",
        queue_name="video.ingestion",
    )

    assert calls.get("publish") is True
    assert "closed" not in calls


def test_open_channel_closes_connection_when_queue_declare_fails(monkeypatch):
    import sys
    import types

    closes = []

    class FakeChannel:
        def queue_declare(self, *, queue, durable):
            raise RuntimeError("queue exists with different config")

    class FakeConnection:
        def channel(self):
            return FakeChannel()

        def close(self):
            closes.append("closed")

    class FakeURLParameters:
        def __init__(self, url):
            self.url = url

    def fake_blocking_connection(params):
        return FakeConnection()

    fake_pika = types.ModuleType("pika")
    fake_pika.BlockingConnection = fake_blocking_connection
    fake_pika.URLParameters = FakeURLParameters
    monkeypatch.setitem(sys.modules, "pika", fake_pika)

    with pytest.raises(RuntimeError, match="queue exists"):
        jobs._open_channel("amqp://broker", "video.ingestion")

    assert closes == ["closed"]


def test_consume_jobs_uses_environment_queue_when_queue_omitted(monkeypatch):
    monkeypatch.setenv("INGESTION_QUEUE", "  env.queue  ")
    calls = {}

    class FakeConnection:
        is_open = True

        def close(self):
            calls["closed"] = True

    class FakeChannel:
        def basic_qos(self, *, prefetch_count):
            calls["prefetch_count"] = prefetch_count

        def basic_consume(self, *, queue, on_message_callback, auto_ack):
            calls["consume"] = {
                "queue": queue,
                "on_message_callback": on_message_callback,
                "auto_ack": auto_ack,
            }

        def start_consuming(self):
            calls["started"] = True

    def fake_open_channel(rabbitmq_url, queue_name):
        calls["open"] = {"rabbitmq_url": rabbitmq_url, "queue_name": queue_name}
        return FakeConnection(), FakeChannel()

    monkeypatch.setattr(jobs, "_open_channel", fake_open_channel)

    jobs.consume_ingestion_jobs(lambda _job: True, rabbitmq_url="amqp://broker")

    assert calls["open"] == {
        "rabbitmq_url": "amqp://broker",
        "queue_name": "env.queue",
    }
    assert calls["prefetch_count"] == 1
    assert calls["consume"]["queue"] == "env.queue"
    assert calls["consume"]["auto_ack"] is False
    assert calls["started"] is True
    assert calls["closed"] is True


def test_consume_jobs_closes_connection_when_consumer_setup_fails(monkeypatch):
    calls = {}

    class FakeConnection:
        is_open = True

        def close(self):
            calls["closed"] = True

    class FakeChannel:
        def basic_qos(self, *, prefetch_count):
            raise RuntimeError("broker refused qos")

    def fake_open_channel(rabbitmq_url, queue_name):
        calls["open"] = {"rabbitmq_url": rabbitmq_url, "queue_name": queue_name}
        return FakeConnection(), FakeChannel()

    monkeypatch.setattr(jobs, "_open_channel", fake_open_channel)

    with pytest.raises(RuntimeError, match="broker refused qos"):
        jobs.consume_ingestion_jobs(
            lambda _job: True,
            rabbitmq_url="amqp://broker",
            queue_name="video.ingestion",
        )

    assert calls["open"] == {
        "rabbitmq_url": "amqp://broker",
        "queue_name": "video.ingestion",
    }
    assert calls["closed"] is True
