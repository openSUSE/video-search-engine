import importlib
import sys
import types


def _load_run_pipeline_with_stubbed_steps(monkeypatch):
    config_module = types.ModuleType("core.config")
    config_module.CONFIG = {
        "filenames": {
            "raw_transcript": "transcript_raw.json",
            "speaker_map": "speaker_map.json",
        },
        "ui": {"host": "127.0.0.1", "port": 5050},
    }
    monkeypatch.setitem(sys.modules, "core.config", config_module)

    stubs = {
        "ingestion_pipeline.steps.step_01_extraction": "run_extraction",
        "ingestion_pipeline.steps.step_02_segmentation": "run_segmentation",
        "ingestion_pipeline.steps.step_03_enrichment": "run_enrichment",
        "ingestion_pipeline.steps.step_04_indexing": "run_indexing",
    }

    for module_name, function_name in stubs.items():
        module = types.ModuleType(module_name)
        setattr(module, function_name, lambda *args, **kwargs: None)
        monkeypatch.setitem(sys.modules, module_name, module)

    sys.modules.pop("ingestion_pipeline.run_pipeline", None)
    return importlib.import_module("ingestion_pipeline.run_pipeline")


def test_external_speaker_map_wait_returns_existing_file(monkeypatch, tmp_path):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    monkeypatch.setenv("SPEAKER_UI_MODE", "external")
    monkeypatch.setenv("SPEAKER_MAP_TIMEOUT_SECONDS", "1")

    speaker_dir = tmp_path / "processed" / "demo"
    speaker_dir.mkdir(parents=True)
    speaker_map = speaker_dir / "speaker_map.json"
    speaker_map.write_text("{}")

    result = run_pipeline.wait_for_speaker_identification(
        str(tmp_path / "demo.mp4"),
        str(tmp_path / "processed"),
    )

    assert result == str(speaker_map)


def test_external_speaker_map_wait_times_out(monkeypatch, tmp_path):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    monkeypatch.setenv("SPEAKER_UI_MODE", "external")
    monkeypatch.setenv("SPEAKER_MAP_TIMEOUT_SECONDS", "0.001")
    monkeypatch.setattr(run_pipeline.time, "sleep", lambda _seconds: None)

    result = run_pipeline.wait_for_speaker_identification(
        str(tmp_path / "demo.mp4"),
        str(tmp_path / "processed"),
    )

    assert result is None
