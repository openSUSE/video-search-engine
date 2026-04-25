import importlib
import logging
import sys
import types

import pytest


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


def test_speaker_ui_mode_strips_blank_environment_values(monkeypatch):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)

    monkeypatch.setenv("SPEAKER_UI_MODE", " local ")
    assert run_pipeline._speaker_ui_mode() == "local"

    monkeypatch.setenv("SPEAKER_UI_MODE", " ")
    assert run_pipeline._speaker_ui_mode() == "external"


def test_blank_speaker_map_timeout_does_not_log_invalid_warning(monkeypatch, caplog):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    monkeypatch.setenv("SPEAKER_MAP_TIMEOUT_SECONDS", " ")

    with caplog.at_level(logging.WARNING):
        assert run_pipeline._speaker_map_timeout_seconds() is None

    assert "Invalid SPEAKER_MAP_TIMEOUT_SECONDS" not in caplog.text


def test_speaker_ui_client_url_uses_loopback_for_wildcard_host(monkeypatch):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)

    assert run_pipeline._speaker_ui_url(
        {"ui": {"host": "0.0.0.0", "port": 5050}}
    ) == "http://127.0.0.1:5050"


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


def test_run_pipeline_rejects_blank_video_path_before_loading_runtime(monkeypatch, tmp_path):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)

    def fail_load_pipeline_steps():
        raise AssertionError("runtime pipeline steps should not load for a blank video path")

    monkeypatch.setattr(run_pipeline, "_load_pipeline_steps", fail_load_pipeline_steps)

    assert not run_pipeline.run_pipeline(" ", str(tmp_path / "processed"))


def test_run_pipeline_main_rejects_blank_video_before_config_and_logging(monkeypatch):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_pipeline", "--video", " "])

    def fail_setup_logging():
        raise AssertionError("setup_logging should not run with a blank video path")

    def fail_load_config():
        raise AssertionError("config should not load before validating required CLI args")

    monkeypatch.setattr(run_pipeline, "setup_logging", fail_setup_logging)
    monkeypatch.setattr(run_pipeline, "_load_config", fail_load_config)

    with pytest.raises(SystemExit) as exc_info:
        run_pipeline.main()

    assert exc_info.value.code == 2


def test_run_pipeline_main_normalizes_string_arguments(monkeypatch, tmp_path):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    default_output_dir = str(tmp_path / "processed")
    monkeypatch.setattr(sys, "argv", [
        "run_pipeline",
        "--video",
        "  data/videos/demo.mp4  ",
        "--output_dir",
        " ",
        "--title",
        "  Demo Title  ",
    ])
    monkeypatch.setattr(run_pipeline, "setup_logging", lambda: None)
    monkeypatch.setattr(
        run_pipeline,
        "_load_config",
        lambda: {"general": {"default_output_dir": default_output_dir}},
    )

    calls = {}

    def fake_run_pipeline(video_path, output_dir, title=None, year=None):
        calls.update(
            {
                "video_path": video_path,
                "output_dir": output_dir,
                "title": title,
                "year": year,
            }
        )
        return True

    monkeypatch.setattr(run_pipeline, "run_pipeline", fake_run_pipeline)

    run_pipeline.main()

    assert calls == {
        "video_path": "data/videos/demo.mp4",
        "output_dir": default_output_dir,
        "title": "Demo Title",
        "year": None,
    }
