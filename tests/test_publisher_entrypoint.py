import sys

import pytest

import ingestion_pipeline.publisher as publisher


def test_publisher_main_normalizes_queue_argument(monkeypatch):
    monkeypatch.setattr(publisher, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(publisher, "setup_logging", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publisher",
            "--video",
            " /data/videos/demo.mp4 ",
            "--rabbitmq-url",
            " amqp://broker ",
            "--queue",
            " custom.queue ",
        ],
    )

    published = {}

    def fake_publish(job, rabbitmq_url, queue_name):
        published["job"] = job
        published["rabbitmq_url"] = rabbitmq_url
        published["queue_name"] = queue_name

    monkeypatch.setattr(publisher, "publish_ingestion_job", fake_publish)

    publisher.main()

    assert published["job"].video_path == "/data/videos/demo.mp4"
    assert published["rabbitmq_url"] == "amqp://broker"
    assert published["queue_name"] == "custom.queue"


def test_publisher_main_rejects_blank_queue_argument(monkeypatch):
    monkeypatch.setattr(publisher, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(publisher, "setup_logging", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publisher",
            "--video",
            "/data/videos/demo.mp4",
            "--rabbitmq-url",
            "amqp://broker",
            "--queue",
            " ",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        publisher.main()

    assert exc_info.value.code == 2


def test_publisher_main_rejects_blank_video_argument_before_logging(monkeypatch):
    monkeypatch.setattr(publisher, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publisher",
            "--video",
            " ",
            "--rabbitmq-url",
            "amqp://broker",
        ],
    )

    def fail_setup_logging():
        raise AssertionError("setup_logging should not run with a blank video path")

    monkeypatch.setattr(publisher, "setup_logging", fail_setup_logging)

    with pytest.raises(SystemExit) as exc_info:
        publisher.main()

    assert exc_info.value.code == 2
