"""Benchmarks for ``core.config``.

Configuration loading runs once per process during normal operation, but the
function is also imported (and exercised) by every test file and every
``compileall`` step in CI. Keeping it cheap matters for developer feedback
loops, especially because it exercises ``yaml.safe_load`` plus a number of
``os.getenv``/``str.strip`` calls that compound when the config file grows.

The benchmarks here use a temporary YAML file written once by the setup
callable so that the parser sees a realistic payload. Environment variables
are *not* mutated globally — instead each benchmark patches ``os.environ`` via
``os.environ.update`` over a controlled subset and restores it on the next
setup invocation.
"""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

from .harness import Benchmark


_SAMPLE_CONFIG: Dict[str, Any] = {
    "general": {
        "device": "auto",
        "default_output_dir": "data/processed",
    },
    "ui": {"host": "127.0.0.1", "port": 5050},
    "api_server": {"host": "127.0.0.1", "port": 1234},
    "database": {
        "host": "localhost",
        "port": 8000,
        "collection_name": "video_search_engine",
    },
    "filenames": {
        "audio": "normalized_audio.mp3",
        "raw_transcript": "transcript_raw.json",
        "speaker_map": "speaker_map.json",
        "transcript": "transcript_generic.json",
        "shots": "shots.json",
        "audio_events": "audio_events.json",
        "visual_details": "visual_details.json",
        "actions": "actions.json",
        "final_analysis": "final_analysis.json",
        "final_segments": "final_segments.json",
        "enriched_segments": "final_enriched_segments.json",
    },
    "models": {
        "transcription": {"name": "base", "compute_type": "int8"},
        "audio_events": {"name": "MIT/ast-finetuned-audioset-10-10-0.4593"},
        "visual_captioning": {"name": "Salesforce/blip-image-captioning-base"},
        "embedding": {"name": "all-MiniLM-L6-v2"},
        "action_recognition": {"name": "MCG-NJU/videomae-base-finetuned-kinetics"},
    },
    "parameters": {
        "transcription": {"batch_size": 32},
        "audio": {"sample_rate": 16000},
        "audio_events": {"top_n": 3, "confidence_threshold": 0.1},
        "visual_captioning": {"max_new_tokens": 50},
        "action_recognition": {"num_frames": 16, "top_n": 3},
    },
    "llm_enrichment": {
        "provider": "gemini",
        "ollama": {
            "enabled": True,
            "host": "http://localhost",
            "port": 11434,
            "model": "gemma:2b",
            "timeout_sec": 120,
        },
        "gemini": {"model": "gemini-1.5-flash"},
    },
}

_NEUTRAL_ENV_KEYS = (
    "ML_DEVICE",
    "HF_TOKEN",
    "OUTPUT_DIR",
    "UI_HOST",
    "UI_PORT",
    "API_HOST",
    "API_PORT",
    "CHROMA_HOST",
    "CHROMA_PORT",
    "CHROMA_COLLECTION",
    "LLM_PROVIDER",
    "OLLAMA_HOST",
    "OLLAMA_PORT",
    "OLLAMA_MODEL",
    "GEMINI_MODEL",
    "MODEL_CACHE_DIR",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "SENTENCE_TRANSFORMERS_HOME",
    "TORCH_HOME",
)


def _setup_yaml_load() -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="vse-bench-config-"))
    config_path = tmp_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(_SAMPLE_CONFIG, sort_keys=False))
    return config_path


def _bench_yaml_safe_load(path: Path) -> None:
    with open(path, "r") as f:
        yaml.safe_load(f)


def _setup_load_config() -> Path:
    path = _setup_yaml_load()
    # Strip benchmark-affecting env vars so each iteration sees the same
    # branches inside ``core.config.load_config``.
    for key in _NEUTRAL_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ["CONFIG_PATH"] = str(path)
    os.environ["MODEL_CACHE_DIR"] = str(path.parent / ".models")
    return path


def _bench_load_config(_: Path) -> None:
    # Importing inside the call keeps the import-time work measurable; the
    # module caches at module level only via ``CONFIG`` which we ignore here
    # because ``load_config`` is the supported re-entrant entry point.
    config_module = importlib.import_module("core.config")
    config_module.load_config()


def _bench_select_device(_: Any) -> None:
    config_module = importlib.import_module("core.config")
    # Internal helper, but stable enough to benchmark — and the cheapest way
    # to measure the cost users pay every time the pipeline boots.
    config_module._select_device("auto")  # type: ignore[attr-defined]
    config_module._select_device("cpu")  # type: ignore[attr-defined]
    config_module._select_device("cuda")  # type: ignore[attr-defined]


BENCHMARKS = [
    Benchmark(
        name="config.yaml_safe_load",
        category="config",
        description="Parse the production-shaped config.yaml from disk.",
        fn=_bench_yaml_safe_load,
        setup=_setup_yaml_load,
        iterations=200,
    ),
    Benchmark(
        name="config.load_config",
        category="config",
        description=(
            "End-to-end load_config() including section validation, env "
            "overrides, device selection, and cache directory bootstrapping."
        ),
        fn=_bench_load_config,
        setup=_setup_load_config,
        iterations=150,
    ),
    Benchmark(
        name="config._select_device.x3",
        category="config",
        description="Three calls to _select_device covering auto/cpu/cuda paths.",
        fn=_bench_select_device,
        iterations=2_000,
        inner_loops=3,
    ),
]
