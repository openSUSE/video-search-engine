import json
import logging
from dataclasses import asdict, dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_QUEUE = "video.ingestion"


@dataclass(frozen=True)
class IngestionJob:
    video_path: str
    output_dir: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.video_path, str) or not self.video_path.strip():
            raise ValueError("video_path must be a non-empty string")
        if self.output_dir is not None and not isinstance(self.output_dir, str):
            raise ValueError("output_dir must be a string")
        if self.title is not None and not isinstance(self.title, str):
            raise ValueError("title must be a string")
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

    return IngestionJob(**payload)


def _open_channel(rabbitmq_url: str, queue_name: str):
    import pika

    connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True)
    return connection, channel


def publish_ingestion_job(job: IngestionJob, rabbitmq_url: str, queue_name: str = DEFAULT_QUEUE) -> None:
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
        connection.close()


def consume_ingestion_jobs(
    handler: Callable[[IngestionJob], bool],
    rabbitmq_url: str,
    queue_name: str = DEFAULT_QUEUE,
) -> None:
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
