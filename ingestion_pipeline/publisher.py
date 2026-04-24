import argparse
import os

from dotenv import load_dotenv

from core.logger import setup_logging
from ingestion_pipeline.jobs import DEFAULT_QUEUE, IngestionJob, publish_ingestion_job


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Publish a video ingestion job to RabbitMQ.")
    parser.add_argument("--video", required=True, help="Path to the video file as seen by the worker.")
    parser.add_argument("--output-dir", help="Optional processed output directory as seen by the worker.")
    parser.add_argument("--title", help="Optional title for metadata lookup and enrichment context.")
    parser.add_argument("--year", type=int, help="Optional release year for metadata lookup.")
    parser.add_argument(
        "--rabbitmq-url",
        default=os.getenv("RABBITMQ_URL", "amqp://video_se:video_se_dev@localhost:5672/%2F"),
        help="RabbitMQ AMQP URL.",
    )
    parser.add_argument(
        "--queue",
        default=os.getenv("INGESTION_QUEUE", DEFAULT_QUEUE),
        help="RabbitMQ queue name.",
    )
    args = parser.parse_args()
    setup_logging()

    job = IngestionJob(
        video_path=args.video,
        output_dir=args.output_dir,
        title=args.title,
        year=args.year,
    )
    publish_ingestion_job(job, rabbitmq_url=args.rabbitmq_url, queue_name=args.queue)


if __name__ == "__main__":
    main()
