import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_QUEUE = "video.ingestion"


def normalize_required_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def normalize_optional_string(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    value = value.strip()
    return value or None


def resolve_rabbitmq_url(rabbitmq_url: Optional[str] = None) -> str:
    resolved_url = os.getenv("RABBITMQ_URL") if rabbitmq_url is None else rabbitmq_url
    return normalize_required_string(resolved_url or "", "RabbitMQ URL")


def resolve_ingestion_queue(queue_name: Optional[str] = None) -> str:
    resolved_queue = (
        os.getenv("INGESTION_QUEUE", DEFAULT_QUEUE) if queue_name is None else queue_name
    )
    return normalize_required_string(resolved_queue, "ingestion queue")


@dataclass(frozen=True)
class IngestionJob:
    video_path: str
    output_dir: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "video_path",
            normalize_required_string(self.video_path, "video_path"),
        )
        object.__setattr__(
            self,
            "output_dir",
            normalize_optional_string(self.output_dir, "output_dir"),
        )
        object.__setattr__(
            self,
            "title",
            normalize_optional_string(self.title, "title"),
        )
        if self.year is not None and type(self.year) is not int:
            raise ValueError("year must be an integer")

    def to_message(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}

    def to_pipeline_kwargs(self, default_output_dir: str) -> dict:
        return {
            "video_path": self.video_path,
            "output_dir": self.output_dir or default_output_dir,
            "title": self.title,
            "year": self.year,
        }


def encode_job_message(job: IngestionJob) -> str:
    return json.dumps(job.to_message(), separators=(",", ":"))


def decode_job_message(body: bytes) -> IngestionJob:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("job message must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("job message must be a JSON object")

    allowed_keys = {"video_path", "output_dir", "title", "year"}
    unexpected_keys = set(payload) - allowed_keys
    if unexpected_keys:
        raise ValueError(f"unexpected job fields: {', '.join(sorted(unexpected_keys))}")
    if "video_path" not in payload:
        raise ValueError("video_path is required")

    return IngestionJob(**payload)


def _open_channel(rabbitmq_url: str, queue_name: str):
    import pika

    connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
    try:
        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=True)
    except Exception:
        # If queue_declare conflicts with an existing queue or the channel
        # setup fails, the BlockingConnection would otherwise leak until the
        # broker idle-times it out. Close it ourselves before re-raising.
        try:
            connection.close()
        except Exception:
            pass
        raise
    return connection, channel


def publish_ingestion_job(
    job: IngestionJob,
    rabbitmq_url: Optional[str],
    queue_name: Optional[str] = None,
) -> None:
    rabbitmq_url = resolve_rabbitmq_url(rabbitmq_url)
    queue_name = resolve_ingestion_queue(queue_name)

    import pika

    connection, channel = _open_channel(rabbitmq_url, queue_name)
    try:
        channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=encode_job_message(job).encode("utf-8"),
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE,
            ),
        )
        logger.info("Published ingestion job for %s to queue %s", job.video_path, queue_name)
    finally:
        if connection.is_open:
            connection.close()


def consume_ingestion_jobs(
    handler: Callable[[IngestionJob], bool],
    rabbitmq_url: Optional[str],
    queue_name: Optional[str] = None,
) -> None:
    rabbitmq_url = resolve_rabbitmq_url(rabbitmq_url)
    queue_name = resolve_ingestion_queue(queue_name)
    connection, channel = _open_channel(rabbitmq_url, queue_name)
    channel.basic_qos(prefetch_count=1)

    def on_message(ch, method, _properties, body):
        try:
            job = decode_job_message(body)
        except ValueError as exc:
            logger.error("Rejecting invalid ingestion job: %s", exc)
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
            return

        logger.info("Starting ingestion job for %s", job.video_path)
        try:
            succeeded = handler(job)
        except Exception:
            logger.exception("Ingestion job failed for %s", job.video_path)
            succeeded = False

        if succeeded:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info("Acknowledged ingestion job for %s", job.video_path)
        else:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            logger.error("Rejected failed ingestion job for %s", job.video_path)

    try:
        channel.basic_consume(queue=queue_name, on_message_callback=on_message, auto_ack=False)
        logger.info("Waiting for ingestion jobs on queue %s", queue_name)
        channel.start_consuming()
    finally:
        if connection.is_open:
            connection.close()
