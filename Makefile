PYTHON ?= python3
COMPOSE ?= docker compose
VENV ?= .venv

.PHONY: venv install-dev test validate publish-ingest compose-up compose-down compose-speaker compose-worker

venv:
	$(PYTHON) -m venv $(VENV)

install-dev: venv
	$(VENV)/bin/python -m pip install -r requirements-dev.txt

test:
	$(VENV)/bin/python -m pytest

validate: test
	$(COMPOSE) config
	$(COMPOSE) --profile worker config
	$(VENV)/bin/python -c "import pathlib, yaml; [list(yaml.safe_load_all(path.read_text())) for path in pathlib.Path('k8s').glob('*.yaml')]"
	@if command -v kubectl >/dev/null 2>&1; then kubectl kustomize k8s >/dev/null; else echo "kubectl not found; skipping kustomize validation"; fi
	$(VENV)/bin/python -m py_compile api/main.py api/schemas.py api/search_utils.py app/main.py app/ui/speaker_id_tool.py core/config.py ingestion_pipeline/jobs.py ingestion_pipeline/publisher.py ingestion_pipeline/worker.py ingestion_pipeline/run_pipeline.py inspect_db.py

publish-ingest:
	@test -n "$(VIDEO)" || (echo "Set VIDEO=/data/videos/your_video.mp4" && exit 1)
	$(VENV)/bin/python -m ingestion_pipeline.publisher --video $(VIDEO)

compose-up:
	$(COMPOSE) up --build chroma rabbitmq api ui-search

compose-speaker:
	$(COMPOSE) up --build ui-speaker

compose-worker:
	$(COMPOSE) --profile worker up --build ingestion-worker

compose-down:
	$(COMPOSE) down
