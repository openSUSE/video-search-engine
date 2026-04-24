PYTHON ?= python3
COMPOSE ?= docker compose
VENV ?= .venv

.PHONY: venv install-dev test compose-up compose-down compose-worker

venv:
	$(PYTHON) -m venv $(VENV)

install-dev: venv
	$(VENV)/bin/python -m pip install -r requirements-dev.txt

test:
	$(VENV)/bin/python -m pytest

compose-up:
	$(COMPOSE) up --build chroma rabbitmq api ui-search

compose-worker:
	$(COMPOSE) --profile worker up --build ingestion-worker

compose-down:
	$(COMPOSE) down
