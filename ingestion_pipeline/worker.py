import os

from core.config import CONFIG
from core.logger import setup_logging
from ingestion_pipeline.jobs import (
    DEFAULT_QUEUE,
    IngestionJob,
    consume_ingestion_jobs,
    resolve_rabbitmq_url,
)


def handle_job(job: IngestionJob) -> bool:
    from ingestion_pipeline.run_pipeline import run_pipeline

    kwargs = job.to_pipeline_kwargs(CONFIG["general"]["default_output_dir"])
    return run_pipeline(**kwargs)


def main() -> None:
    setup_logging()
    try:
        rabbitmq_url = resolve_rabbitmq_url()
    except ValueError as exc:
        raise SystemExit(str(exc))
    queue_name = os.getenv("INGESTION_QUEUE", DEFAULT_QUEUE)
    consume_ingestion_jobs(handle_job, rabbitmq_url=rabbitmq_url, queue_name=queue_name)


if __name__ == "__main__":
    main()
