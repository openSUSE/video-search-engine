import importlib
import json
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
    (speaker_dir / "transcript_raw.json").write_text(
        json.dumps([{"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "hello"}])
    )
    speaker_map = speaker_dir / "speaker_map.json"
    speaker_map.write_text(json.dumps({"SPEAKER_00": "Alice"}))

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


def test_external_speaker_map_wait_requires_all_transcript_speakers(monkeypatch, tmp_path):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    monkeypatch.setenv("SPEAKER_UI_MODE", "external")
    monkeypatch.setenv("SPEAKER_MAP_TIMEOUT_SECONDS", "0.001")
    monkeypatch.setattr(run_pipeline.time, "sleep", lambda _seconds: None)

    speaker_dir = tmp_path / "processed" / "demo"
    speaker_dir.mkdir(parents=True)
    (speaker_dir / "transcript_raw.json").write_text(
        json.dumps(
            [
                {"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "hello"},
                {"start": 1, "end": 2, "speaker": "SPEAKER_01", "text": "reply"},
            ]
        )
    )
    (speaker_dir / "speaker_map.json").write_text(
        json.dumps({"SPEAKER_00": "Alice"})
    )

    result = run_pipeline.wait_for_speaker_identification(
        str(tmp_path / "demo.mp4"),
        str(tmp_path / "processed"),
    )

    assert result is None


def test_speaker_map_readiness_allows_empty_map_for_transcript_without_speakers(
    monkeypatch,
    tmp_path,
):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    transcript_path = tmp_path / "transcript_raw.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    transcript_path.write_text(json.dumps([{"start": 0, "end": 1, "text": "music"}]))
    speaker_map_path.write_text("{}")

    assert run_pipeline._speaker_map_readiness(
        str(speaker_map_path),
        str(transcript_path),
    ) == (True, None)


def test_external_speaker_map_wait_creates_empty_map_for_transcript_without_speakers(
    monkeypatch,
    tmp_path,
):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    monkeypatch.setenv("SPEAKER_UI_MODE", "external")
    monkeypatch.setenv("SPEAKER_MAP_TIMEOUT_SECONDS", "0.001")
    monkeypatch.setattr(run_pipeline.time, "sleep", lambda _seconds: None)

    speaker_dir = tmp_path / "processed" / "demo"
    speaker_dir.mkdir(parents=True)
    (speaker_dir / "transcript_raw.json").write_text(
        json.dumps([{"start": 0, "end": 1, "text": "music"}])
    )
    speaker_map = speaker_dir / "speaker_map.json"

    result = run_pipeline.wait_for_speaker_identification(
        str(tmp_path / "demo.mp4"),
        str(tmp_path / "processed"),
    )

    assert result == str(speaker_map)
    assert speaker_map.read_text() == "{}"


def test_speaker_map_readiness_rejects_malformed_transcript(
    monkeypatch,
    tmp_path,
):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    transcript_path = tmp_path / "transcript_raw.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    transcript_path.write_text(json.dumps([{"speaker": "SPEAKER_00", "text": "hello"}]))
    speaker_map_path.write_text(json.dumps({"SPEAKER_00": "Alice"}))

    is_ready, wait_reason = run_pipeline._speaker_map_readiness(
        str(speaker_map_path),
        str(transcript_path),
    )

    assert is_ready is False
    assert "start" in wait_reason


def test_speaker_map_wait_fails_fast_for_malformed_transcript(
    monkeypatch,
    tmp_path,
    caplog,
):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    transcript_path = tmp_path / "transcript_raw.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    transcript_path.write_text(json.dumps([{"speaker": "SPEAKER_00", "text": "hello"}]))
    speaker_map_path.write_text(json.dumps({"SPEAKER_00": "Alice"}))

    def fail_sleep(_seconds):
        raise AssertionError("malformed transcript should not keep the wait loop alive")

    monkeypatch.setattr(run_pipeline.time, "sleep", fail_sleep)

    with caplog.at_level(logging.ERROR):
        result = run_pipeline._wait_until_speaker_map_ready(
            str(speaker_map_path),
            str(transcript_path),
        )

    assert result is False
    assert "malformed raw transcript" in caplog.text


def test_local_speaker_mode_completes_no_speaker_transcript_without_launching_ui(
    monkeypatch,
    tmp_path,
):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    monkeypatch.setenv("SPEAKER_UI_MODE", "local")

    speaker_dir = tmp_path / "processed" / "demo"
    speaker_dir.mkdir(parents=True)
    (speaker_dir / "transcript_raw.json").write_text(
        json.dumps([{"start": 0, "end": 1, "text": "music"}])
    )
    speaker_map = speaker_dir / "speaker_map.json"
    calls = {}

    def fake_popen(_command):
        calls["popen"] = True
        raise AssertionError(
            "local speaker UI should not start for no-speaker transcript"
        )

    monkeypatch.setattr(run_pipeline.subprocess, "Popen", fake_popen)

    result = run_pipeline.wait_for_speaker_identification(
        str(tmp_path / "demo.mp4"),
        str(tmp_path / "processed"),
    )

    assert result == str(speaker_map)
    assert speaker_map.read_text() == "{}"
    assert "popen" not in calls


def test_local_speaker_mode_fails_malformed_transcript_before_launching_ui(
    monkeypatch,
    tmp_path,
):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    monkeypatch.setenv("SPEAKER_UI_MODE", "local")

    speaker_dir = tmp_path / "processed" / "demo"
    speaker_dir.mkdir(parents=True)
    (speaker_dir / "transcript_raw.json").write_text(
        json.dumps([{"speaker": "SPEAKER_00", "text": "hello"}])
    )
    calls = {}

    def fake_popen(_command):
        calls["popen"] = True
        raise AssertionError(
            "local speaker UI should not start for malformed transcript"
        )

    monkeypatch.setattr(run_pipeline.subprocess, "Popen", fake_popen)

    result = run_pipeline.wait_for_speaker_identification(
        str(tmp_path / "demo.mp4"),
        str(tmp_path / "processed"),
    )

    assert result is None
    assert "popen" not in calls


def test_local_speaker_mode_does_not_skip_existing_partial_map(monkeypatch, tmp_path):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    monkeypatch.setenv("SPEAKER_UI_MODE", "local")

    speaker_dir = tmp_path / "processed" / "demo"
    speaker_dir.mkdir(parents=True)
    (speaker_dir / "transcript_raw.json").write_text(
        json.dumps(
            [
                {"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "hello"},
                {"start": 1, "end": 2, "speaker": "SPEAKER_01", "text": "reply"},
            ]
        )
    )
    (speaker_dir / "speaker_map.json").write_text(
        json.dumps({"SPEAKER_00": "Alice"})
    )
    calls = {}

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            calls["terminated"] = True

        def wait(self, timeout):
            calls["wait_timeout"] = timeout

    def fake_popen(command):
        calls["command"] = command
        return FakeProcess()

    monkeypatch.setattr(run_pipeline.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_pipeline, "_wait_until_speaker_map_ready", lambda *args, **kwargs: False)

    result = run_pipeline.wait_for_speaker_identification(
        str(tmp_path / "demo.mp4"),
        str(tmp_path / "processed"),
    )

    assert result is None
    assert calls["command"][:3] == [sys.executable, "-m", "app.main"]
    assert calls["terminated"] is True


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


@pytest.mark.parametrize("raw_timeout", ["nan", "inf", "-inf"])
def test_speaker_map_timeout_rejects_nonfinite_values(
    monkeypatch,
    caplog,
    raw_timeout,
):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    monkeypatch.setenv("SPEAKER_MAP_TIMEOUT_SECONDS", raw_timeout)

    with caplog.at_level(logging.WARNING):
        assert run_pipeline._speaker_map_timeout_seconds() is None

    assert "Invalid SPEAKER_MAP_TIMEOUT_SECONDS" in caplog.text


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

    analysis_path = tmp_path / "analysis.json"
    extraction_module = types.ModuleType("ingestion_pipeline.steps.step_01_extraction")

    def fake_module_run_extraction(*args, **kwargs):
        calls["extraction_config"] = kwargs.get("config")
        analysis_path.write_text("[]")
        return str(analysis_path)

    extraction_module.run_extraction = fake_module_run_extraction
    monkeypatch.setitem(
        sys.modules,
        "ingestion_pipeline.steps.step_01_extraction",
        extraction_module,
    )

    segmentation_module = types.ModuleType("ingestion_pipeline.steps.step_02_segmentation")

    def fake_run_segmentation(**kwargs):
        calls["segmentation_config"] = kwargs.get("config")
        calls["segmentation_analysis_path"] = kwargs.get("analysis_path")
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
    (speaker_dir / "transcript_raw.json").write_text(
        json.dumps([{"start": 0, "end": 1, "text": "music"}])
    )
    (speaker_dir / "speaker_map.json").write_text("{}")

    assert run_pipeline.run_pipeline(str(tmp_path / "demo.mp4"), str(output_dir))
    assert calls == {
        "extraction_config": config,
        "segmentation_config": config,
        "segmentation_analysis_path": str(analysis_path),
        "enrichment_config": config,
        "indexing_config": config,
    }


def test_run_pipeline_uses_injected_config_without_reloading(monkeypatch, tmp_path):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    config = {
        "filenames": {
            "raw_transcript": "transcript_raw.json",
            "speaker_map": "speaker_map.json",
            "final_analysis": "analysis.json",
        }
    }
    calls = {}

    def fail_load_config():
        raise AssertionError("injected config should be reused")

    def fake_run_extraction(*args, **kwargs):
        calls["extraction_config"] = kwargs.get("config")
        analysis_path = tmp_path / "analysis.json"
        analysis_path.write_text("[]")
        return str(analysis_path)

    def fake_run_segmentation(**kwargs):
        calls["segmentation_config"] = kwargs.get("config")
        return str(tmp_path / "segments.json")

    def fake_run_enrichment(_segments_path, received_config):
        calls["enrichment_config"] = received_config
        return str(tmp_path / "enriched.json")

    def fake_run_indexing(**kwargs):
        calls["indexing_config"] = kwargs.get("config")

    def fake_wait_for_speaker_identification(_video_path, _output_dir, config=None):
        calls["speaker_wait_config"] = config
        return str(tmp_path / "speaker_map.json")

    monkeypatch.setattr(run_pipeline, "_load_config", fail_load_config)
    monkeypatch.setattr(
        run_pipeline,
        "_load_pipeline_steps",
        lambda: (
            fake_run_extraction,
            fake_run_segmentation,
            fake_run_enrichment,
            fake_run_indexing,
        ),
    )
    monkeypatch.setattr(
        run_pipeline,
        "wait_for_speaker_identification",
        fake_wait_for_speaker_identification,
    )

    assert run_pipeline.run_pipeline(
        str(tmp_path / "demo.mp4"),
        str(tmp_path / "processed"),
        config=config,
    )
    assert calls == {
        "extraction_config": config,
        "speaker_wait_config": config,
        "segmentation_config": config,
        "enrichment_config": config,
        "indexing_config": config,
    }


def test_run_pipeline_halts_before_indexing_when_enrichment_fails(
    monkeypatch,
    tmp_path,
):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    config = {
        "filenames": {
            "raw_transcript": "transcript_raw.json",
            "speaker_map": "speaker_map.json",
            "final_analysis": "analysis.json",
        }
    }
    calls = []

    def fake_run_extraction(*_args, **_kwargs):
        calls.append("extraction")
        analysis_path = tmp_path / "analysis.json"
        analysis_path.write_text("[]")
        return str(analysis_path)

    def fake_run_segmentation(**_kwargs):
        calls.append("segmentation")
        return str(tmp_path / "segments.json")

    def fake_run_enrichment(_segments_path, _config):
        calls.append("enrichment")
        return None

    def fail_run_indexing(**_kwargs):
        calls.append("indexing")
        raise AssertionError("indexing should not run after failed enrichment")

    monkeypatch.setattr(
        run_pipeline,
        "_load_pipeline_steps",
        lambda: (
            fake_run_extraction,
            fake_run_segmentation,
            fake_run_enrichment,
            fail_run_indexing,
        ),
    )
    monkeypatch.setattr(
        run_pipeline,
        "wait_for_speaker_identification",
        lambda *_args, **_kwargs: str(tmp_path / "speaker_map.json"),
    )

    assert not run_pipeline.run_pipeline(
        str(tmp_path / "demo.mp4"),
        str(tmp_path / "processed"),
        config=config,
    )
    assert calls == ["extraction", "segmentation", "enrichment"]


@pytest.mark.parametrize("return_missing_path", [False, True])
def test_run_pipeline_halts_before_speaker_wait_when_extraction_fails(
    monkeypatch,
    tmp_path,
    return_missing_path,
):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    config = {
        "filenames": {
            "raw_transcript": "transcript_raw.json",
            "speaker_map": "speaker_map.json",
            "final_analysis": "analysis.json",
        }
    }
    calls = []

    def fake_run_extraction(*_args, **_kwargs):
        calls.append("extraction")
        if return_missing_path:
            return str(tmp_path / "missing-analysis.json")
        return None

    def fail_wait_for_speaker_identification(*_args, **_kwargs):
        calls.append("speaker_wait")
        raise AssertionError("speaker wait should not run after failed extraction")

    def fail_run_segmentation(**_kwargs):
        calls.append("segmentation")
        raise AssertionError("segmentation should not run after failed extraction")

    monkeypatch.setattr(
        run_pipeline,
        "_load_pipeline_steps",
        lambda: (
            fake_run_extraction,
            fail_run_segmentation,
            lambda *_args, **_kwargs: str(tmp_path / "enriched.json"),
            lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(
        run_pipeline,
        "wait_for_speaker_identification",
        fail_wait_for_speaker_identification,
    )

    assert not run_pipeline.run_pipeline(
        str(tmp_path / "demo.mp4"),
        str(tmp_path / "processed"),
        config=config,
    )
    assert calls == ["extraction"]


def test_run_pipeline_rejects_blank_video_path_before_loading_runtime(monkeypatch, tmp_path):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)

    def fail_load_pipeline_steps():
        raise AssertionError("runtime pipeline steps should not load for a blank video path")

    monkeypatch.setattr(run_pipeline, "_load_pipeline_steps", fail_load_pipeline_steps)

    assert not run_pipeline.run_pipeline(" ", str(tmp_path / "processed"))


@pytest.mark.parametrize("year", [0, -1, True])
def test_run_pipeline_rejects_invalid_year_before_loading_runtime(
    monkeypatch,
    tmp_path,
    year,
):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)

    def fail_load_pipeline_steps():
        raise AssertionError("runtime pipeline steps should not load for an invalid year")

    monkeypatch.setattr(run_pipeline, "_load_pipeline_steps", fail_load_pipeline_steps)

    assert not run_pipeline.run_pipeline(
        str(tmp_path / "demo.mp4"),
        str(tmp_path / "processed"),
        year=year,
    )


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


def test_run_pipeline_main_rejects_invalid_year_before_config_and_logging(monkeypatch):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_pipeline", "--video", "demo.mp4", "--year", "0"])

    def fail_setup_logging():
        raise AssertionError("setup_logging should not run with an invalid year")

    def fail_load_config():
        raise AssertionError("config should not load before validating CLI args")

    monkeypatch.setattr(run_pipeline, "setup_logging", fail_setup_logging)
    monkeypatch.setattr(run_pipeline, "_load_config", fail_load_config)

    with pytest.raises(SystemExit) as exc_info:
        run_pipeline.main()

    assert exc_info.value.code == 2


def test_run_pipeline_main_normalizes_string_arguments(monkeypatch, tmp_path):
    run_pipeline = _load_run_pipeline_with_stubbed_steps(monkeypatch)
    default_output_dir = str(tmp_path / "processed")
    loaded_config = {"general": {"default_output_dir": default_output_dir}}
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
        lambda: loaded_config,
    )

    calls = {}

    def fake_run_pipeline(video_path, output_dir, title=None, year=None, config=None):
        calls.update(
            {
                "video_path": video_path,
                "output_dir": output_dir,
                "title": title,
                "year": year,
                "config": config,
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
        "config": loaded_config,
    }
