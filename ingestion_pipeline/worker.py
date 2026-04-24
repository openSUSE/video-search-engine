import os

from core.config import CONFIG
from core.logger import setup_logging
from ingestion_pipeline.jobs import DEFAULT_QUEUE, IngestionJob, consume_ingestion_jobs


def handle_job(job: IngestionJob) -> bool:
    from ingestion_pipeline.run_pipeline import run_pipeline

    kwargs = job.to_pipeline_kwargs(CONFIG["general"]["default_output_dir"])
    return run_pipeline(**kwargs)


def main() -> None:
    setup_logging()
    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://video_se:video_se_dev@localhost:5672/%2F")
    queue_name = os.getenv("INGESTION_QUEUE", DEFAULT_QUEUE)
    consume_ingestion_jobs(handle_job, rabbitmq_url=rabbitmq_url, queue_name=queue_name)


if __name__ == "__main__":
    main()
