PYTHON ?= python3
COMPOSE ?= docker compose
VENV ?= .venv

.PHONY: venv install-dev test test-integration validate audit bench bench-smoke bench-baseline bench-check publish-ingest compose-up compose-down compose-speaker compose-worker

venv:
	$(PYTHON) -m venv $(VENV)

install-dev: venv
	$(VENV)/bin/python -m pip install -r requirements-dev.txt

test:
	$(VENV)/bin/python -m pytest -m "not integration"

test-integration:
	$(VENV)/bin/python -m pytest tests/integration -v

# Scan pinned api/dev/ui requirements for known CVEs. Heavy ingestion
# deps (torch, whisperx) are excluded — their advisories are noisy and
# the worker is internet-isolated in our deployment shape. Override
# AUDIT_FILES if you want to include them.
AUDIT_FILES ?= requirements-api.txt requirements-dev.txt requirements-ui.txt
audit:
	@command -v $(VENV)/bin/pip-audit >/dev/null 2>&1 || $(VENV)/bin/pip install pip-audit==2.7.3
	@for f in $(AUDIT_FILES); do \
		echo "==> auditing $$f"; \
		$(VENV)/bin/pip-audit -r $$f --strict || exit 1; \
	done

validate: test
	$(COMPOSE) config
	$(COMPOSE) --profile worker config
	$(VENV)/bin/python -c "import pathlib, yaml; [list(yaml.safe_load_all(path.read_text())) for path in pathlib.Path('k8s').glob('*.yaml')]"
	@if command -v kubectl >/dev/null 2>&1; then kubectl kustomize k8s >/dev/null; else echo "kubectl not found; skipping kustomize validation"; fi
	$(VENV)/bin/python -m compileall -q api app core ingestion_pipeline inspect_db.py

bench:
	$(VENV)/bin/python -m benchmarks.runner

bench-smoke:
	$(VENV)/bin/python -m benchmarks.runner --scale 0.1 --quiet

# Capture a baseline JSON snapshot for trend tracking. The path is configurable
# so callers can pin the baseline per-branch or per-environment.
BASELINE ?= benchmarks/reports/baseline.json
bench-baseline:
	@mkdir -p $(dir $(BASELINE))
	$(VENV)/bin/python -m benchmarks.runner --json $(BASELINE)

# Compare the current code against a previously captured baseline, gating on
# regressions past WARN_RATIO (default 0.15 — sub-microsecond benchmarks
# regularly jitter ±10% from CPU/GC noise alone, so 0.15 catches real
# regressions while filtering out scheduler hiccups).
WARN_RATIO ?= 0.15
bench-check:
	@test -f $(BASELINE) || (echo "$(BASELINE) not found — run 'make bench-baseline' first" && exit 1)
	$(VENV)/bin/python -m benchmarks.runner \
		--baseline $(BASELINE) \
		--warn-ratio $(WARN_RATIO) \
		--fail-on-regression \
		--quiet

publish-ingest:
	@test -n "$(VIDEO)" || (echo "Set VIDEO=/data/videos/your_video.mp4" && exit 1)
	$(VENV)/bin/python -m ingestion_pipeline.publisher --video "$(VIDEO)"

compose-up:
	$(COMPOSE) up --build chroma rabbitmq api ui-search

compose-speaker:
	$(COMPOSE) up --build ui-speaker

compose-worker:
	$(COMPOSE) --profile worker up --build ingestion-worker

compose-down:
	$(COMPOSE) down
