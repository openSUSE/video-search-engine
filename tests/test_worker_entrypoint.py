import pytest

import ingestion_pipeline.worker as worker


def test_worker_main_loads_rabbitmq_url_from_dotenv(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("RABBITMQ_URL=amqp://from-dotenv\nINGESTION_QUEUE=dotenv-queue\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RABBITMQ_URL", raising=False)
    monkeypatch.delenv("INGESTION_QUEUE", raising=False)
    monkeypatch.setattr(worker, "setup_logging", lambda: None)

    consumed = {}

    def fake_consume(handler, rabbitmq_url, queue_name):
        consumed["handler"] = handler
        consumed["rabbitmq_url"] = rabbitmq_url
        consumed["queue_name"] = queue_name

    monkeypatch.setattr(worker, "consume_ingestion_jobs", fake_consume)

    worker.main()

    assert consumed["handler"] is worker.handle_job
    assert consumed["rabbitmq_url"] == "amqp://from-dotenv"
    assert consumed["queue_name"] == "dotenv-queue"


def test_worker_main_exits_before_logging_without_rabbitmq_url(monkeypatch):
    monkeypatch.delenv("RABBITMQ_URL", raising=False)

    def fail_setup_logging():
        raise AssertionError("setup_logging should not run without RABBITMQ_URL")

    monkeypatch.setattr(worker, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "setup_logging", fail_setup_logging)

    with pytest.raises(SystemExit, match="RabbitMQ URL"):
        worker.main()
