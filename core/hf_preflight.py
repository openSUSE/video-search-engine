"""HuggingFace access preflight for the ingestion entrypoints.

Goal: tell the operator *before* whisperx tries to load the diarization
model whether their HF setup is going to actually work. Three failure
modes today:

* **HF_TOKEN unset** — every HF download fails with a generic 401 deep
  inside whisperx; the user has no idea why.
* **HF_TOKEN set but invalid / expired** — same 401, same confusion.
* **HF_TOKEN valid but the user hasn't accepted the gated terms for
  ``pyannote/speaker-diarization-community-1`` and
  ``pyannote/segmentation-3.0``** — looks like a model-not-found error
  unless you read the URL.

This module sends one ``HEAD`` request per gated model and turns the
result into a single, copy-pastable status line. It is purely
informational: the worker / run_pipeline keep going regardless, since
``transcribe_and_diarize`` has a graceful fallback that produces a
transcript without speaker labels when diarization is unreachable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, List

logger = logging.getLogger(__name__)

GATED_MODELS = (
    "pyannote/speaker-diarization-community-1",
    "pyannote/segmentation-3.0",
)
MODEL_CONFIG_URL_TEMPLATE = (
    "https://huggingface.co/{model}/resolve/main/config.yaml"
)
MODEL_TERMS_URL_TEMPLATE = "https://hf.co/{model}"
PREFLIGHT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class ModelAccessResult:
    model: str
    status: str  # "ok" | "missing_token" | "invalid_token" | "needs_terms" | "unreachable"
    detail: str = ""


@dataclass(frozen=True)
class PreflightReport:
    models: List[ModelAccessResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return bool(self.models) and all(m.status == "ok" for m in self.models)

    @property
    def has_token(self) -> bool:
        return all(m.status != "missing_token" for m in self.models)

    @property
    def needs_terms(self) -> List[ModelAccessResult]:
        return [m for m in self.models if m.status == "needs_terms"]


def _check_one_model(model: str, token: str) -> ModelAccessResult:
    if not token:
        return ModelAccessResult(
            model=model,
            status="missing_token",
            detail="HF_TOKEN is not set in the environment.",
        )
    try:
        import requests
    except ImportError:
        return ModelAccessResult(
            model=model,
            status="unreachable",
            detail="`requests` is not installed; cannot probe HuggingFace.",
        )
    url = MODEL_CONFIG_URL_TEMPLATE.format(model=model)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.head(
            url,
            headers=headers,
            timeout=PREFLIGHT_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        return ModelAccessResult(
            model=model,
            status="unreachable",
            detail=f"Network error reaching {url}: {exc}",
        )
    if response.status_code == 200:
        return ModelAccessResult(model=model, status="ok")
    if response.status_code == 401:
        return ModelAccessResult(
            model=model,
            status="invalid_token",
            detail=(
                "HuggingFace returned 401 — your HF_TOKEN is invalid or "
                "expired. Generate a new token at "
                "https://huggingface.co/settings/tokens."
            ),
        )
    if response.status_code == 403:
        return ModelAccessResult(
            model=model,
            status="needs_terms",
            detail=(
                f"HuggingFace returned 403 — visit "
                f"{MODEL_TERMS_URL_TEMPLATE.format(model=model)} and click "
                "'Agree and access repository' to grant your token access."
            ),
        )
    return ModelAccessResult(
        model=model,
        status="unreachable",
        detail=(
            f"HuggingFace returned unexpected status {response.status_code} "
            f"for {url}."
        ),
    )


def check_huggingface_access(
    *,
    token: str | None = None,
    models: Iterable[str] = GATED_MODELS,
) -> PreflightReport:
    """Probe each gated model and return a structured report.

    Token defaults to the ``HF_TOKEN`` env var. Models default to the
    set whisperx requires.
    """
    if token is None:
        token = (os.environ.get("HF_TOKEN") or "").strip()
    results = [_check_one_model(model, token) for model in models]
    return PreflightReport(models=results)


def log_preflight_report(report: PreflightReport) -> None:
    """Render the report to logs in a format that's both human-readable
    and easy to spot in JSON-mode log output.

    Status priority (worst case first):
    * any ``invalid_token`` → ERROR with rotation instructions
    * any ``missing_token`` → WARNING with link to settings
    * any ``needs_terms`` → WARNING with the exact URL to click
    * any ``unreachable`` → INFO (transient; runtime will retry)
    * all ``ok`` → INFO with a single happy line
    """
    if report.all_ok:
        logger.info(
            "HuggingFace preflight: all gated models accessible (%s).",
            ", ".join(m.model for m in report.models),
        )
        return

    by_status = {
        m.status: m for m in report.models
    }  # latest model wins per status, fine for messaging

    if any(m.status == "invalid_token" for m in report.models):
        logger.error(
            "HuggingFace preflight: HF_TOKEN appears invalid. "
            "Diarization will be skipped at runtime (transcripts will "
            "have no speaker labels). Rotate the token at "
            "https://huggingface.co/settings/tokens and update HF_TOKEN."
        )
    elif any(m.status == "missing_token" for m in report.models):
        logger.warning(
            "HuggingFace preflight: HF_TOKEN is not set. Diarization will "
            "be skipped — set HF_TOKEN in the environment to enable it."
        )

    needs_terms = [m for m in report.models if m.status == "needs_terms"]
    if needs_terms:
        urls = "\n  - ".join(
            MODEL_TERMS_URL_TEMPLATE.format(model=m.model) for m in needs_terms
        )
        logger.warning(
            "HuggingFace preflight: your HF_TOKEN does not have access to "
            "%d gated model(s) used by whisperx diarization. Diarization "
            "will be skipped at runtime — accept the terms here:\n  - %s",
            len(needs_terms),
            urls,
        )

    unreachable = [m for m in report.models if m.status == "unreachable"]
    if unreachable and not (
        any(m.status == "invalid_token" for m in report.models)
        or any(m.status == "missing_token" for m in report.models)
        or needs_terms
    ):
        logger.info(
            "HuggingFace preflight: could not reach %d model(s) — runtime "
            "will retry. (%s)",
            len(unreachable),
            "; ".join(f"{m.model}: {m.detail}" for m in unreachable),
        )
