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


def test_run_pipeline_threads_loaded_config_into_segmentation(monkeypatch, tmp_path):
    config = {
        "filenames": {
            "raw_transcript": "transcript_raw.json",
            "speaker_map": "speaker_map.json",
            "final_analysis": "analysis.json",
        }
    }
    config_module = types.ModuleType("core.config")
    config_module.CONFIG = config
    monkeypatch.setitem(sys.modules, "core.config", config_module)

    calls = {}

    extraction_module = types.ModuleType("ingestion_pipeline.steps.step_01_extraction")
    extraction_module.run_extraction = lambda *args, **kwargs: calls.setdefault(
        "extraction_config",
        kwargs.get("config"),
    )
    monkeypatch.setitem(
        sys.modules,
        "ingestion_pipeline.steps.step_01_extraction",
        extraction_module,
    )

    segmentation_module = types.ModuleType("ingestion_pipeline.steps.step_02_segmentation")

    def fake_run_segmentation(**kwargs):
        calls["segmentation_config"] = kwargs.get("config")
        return str(tmp_path / "segments.json")

    segmentation_module.run_segmentation = fake_run_segmentation
    monkeypatch.setitem(
        sys.modules,
        "ingestion_pipeline.steps.step_02_segmentation",
        segmentation_module,
    )

    enrichment_module = types.ModuleType("ingestion_pipeline.steps.step_03_enrichment")

    def fake_run_enrichment(_segments_path, received_config):
        calls["enrichment_config"] = received_config
        return str(tmp_path / "enriched.json")

    enrichment_module.run_enrichment = fake_run_enrichment
    monkeypatch.setitem(
        sys.modules,
        "ingestion_pipeline.steps.step_03_enrichment",
        enrichment_module,
    )

    indexing_module = types.ModuleType("ingestion_pipeline.steps.step_04_indexing")
    indexing_module.run_indexing = lambda **kwargs: calls.setdefault(
        "indexing_config",
        kwargs.get("config"),
    )
    monkeypatch.setitem(
        sys.modules,
        "ingestion_pipeline.steps.step_04_indexing",
        indexing_module,
    )

    sys.modules.pop("ingestion_pipeline.run_pipeline", None)
    run_pipeline = importlib.import_module("ingestion_pipeline.run_pipeline")
    monkeypatch.setenv("SPEAKER_UI_MODE", "external")

    output_dir = tmp_path / "processed"
    speaker_dir = output_dir / "demo"
    speaker_dir.mkdir(parents=True)
    (speaker_dir / "speaker_map.json").write_text("{}")

    assert run_pipeline.run_pipeline(str(tmp_path / "demo.mp4"), str(output_dir))
    assert calls == {
        "extraction_config": config,
        "segmentation_config": config,
        "enrichment_config": config,
        "indexing_config": config,
    }
