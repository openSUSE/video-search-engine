import builtins
import json

import pytest

from ingestion_pipeline.steps.step_03_enrichment import _call_gemini_api, run_enrichment


def test_run_enrichment_uses_injected_provider_and_preserves_completed_segments(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "full_transcript": "important dialogue",
                    "speakers": ["Alice"],
                    "consolidated_visual_captions": ["a train platform"],
                    "consolidated_actions": ["waiting"],
                    "consolidated_audio_events": ["announcement"],
                },
                {
                    "segment_id": "segment_0002",
                    "title": "Already Done",
                    "summary": "Existing summary",
                    "keywords": ["existing"],
                    "full_transcript": "skip me",
                },
            ]
        )
    )
    (tmp_path / "video_metadata.json").write_text(
        json.dumps({"title": "Demo Movie", "synopsis": "A station scene."})
    )
    calls = []

    def fake_ollama_client(prompt, config):
        calls.append({"prompt": prompt, "config": config})
        return {
            "title": "Generated Title",
            "summary": "Generated summary.",
            "keywords": ["platform", "announcement"],
        }

    config = {
        "filenames": {"enriched_segments": "enriched.json"},
        "llm_enrichment": {"provider": "ollama"},
    }

    output_path = run_enrichment(
        str(segments_path),
        config,
        llm_clients={"ollama": fake_ollama_client},
    )

    assert output_path == str(tmp_path / "enriched.json")
    enriched_segments = json.loads((tmp_path / "enriched.json").read_text())
    assert enriched_segments[0]["title"] == "Generated Title"
    assert enriched_segments[0]["keywords"] == ["platform", "announcement"]
    assert enriched_segments[1]["title"] == "Already Done"
    assert len(calls) == 1
    assert "Demo Movie" in calls[0]["prompt"]
    assert "important dialogue" in calls[0]["prompt"]
    assert calls[0]["config"] == config


def test_run_enrichment_normalizes_provider_name(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "full_transcript": "dialogue",
                    "speakers": [],
                    "consolidated_visual_captions": [],
                    "consolidated_actions": [],
                    "consolidated_audio_events": [],
                }
            ]
        )
    )
    calls = []

    def fake_ollama_client(prompt, _config):
        calls.append(prompt)
        return {"title": "Generated", "summary": "Summary.", "keywords": ["demo"]}

    output_path = run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": " OLLAMA "},
        },
        llm_clients={"ollama": fake_ollama_client},
    )

    assert output_path == str(tmp_path / "enriched.json")
    assert len(calls) == 1


def test_run_enrichment_uses_logline_metadata_as_synopsis(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "full_transcript": "dialogue",
                    "speakers": [],
                    "consolidated_visual_captions": [],
                    "consolidated_actions": [],
                    "consolidated_audio_events": [],
                }
            ]
        )
    )
    (tmp_path / "video_metadata.json").write_text(
        json.dumps({"title": "Demo Movie", "logline": "A legacy overview."})
    )
    calls = []

    def fake_ollama_client(prompt, config):
        calls.append(prompt)
        return {"title": "Generated", "summary": "Summary.", "keywords": ["demo"]}

    run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": "ollama"},
        },
        llm_clients={"ollama": fake_ollama_client},
    )

    assert "A legacy overview." in calls[0]


def test_run_enrichment_ignores_non_object_video_metadata(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "full_transcript": "dialogue",
                    "speakers": [],
                    "consolidated_visual_captions": [],
                    "consolidated_actions": [],
                    "consolidated_audio_events": [],
                }
            ]
        )
    )
    (tmp_path / "video_metadata.json").write_text(json.dumps(["not", "metadata"]))
    calls = []

    def fake_ollama_client(prompt, _config):
        calls.append(prompt)
        return {"title": "Generated", "summary": "Summary.", "keywords": ["demo"]}

    output_path = run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": "ollama"},
        },
        llm_clients={"ollama": fake_ollama_client},
    )

    assert output_path == str(tmp_path / "enriched.json")
    assert "- Title: N/A" in calls[0]
    assert "- Synopsis: N/A" in calls[0]


def test_run_enrichment_retries_partially_enriched_segments(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "title": "Only Title Exists",
                    "full_transcript": "dialogue",
                    "speakers": [],
                    "consolidated_visual_captions": [],
                    "consolidated_actions": [],
                    "consolidated_audio_events": [],
                }
            ]
        )
    )
    calls = []

    def fake_ollama_client(prompt, _config):
        calls.append(prompt)
        return {
            "title": "Regenerated Title",
            "summary": "Regenerated summary.",
            "keywords": ["complete"],
        }

    run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": "ollama"},
        },
        llm_clients={"ollama": fake_ollama_client},
    )

    [segment] = json.loads((tmp_path / "enriched.json").read_text())
    assert len(calls) == 1
    assert segment["title"] == "Regenerated Title"
    assert segment["summary"] == "Regenerated summary."
    assert segment["keywords"] == ["complete"]


def test_run_enrichment_does_not_apply_structural_llm_fields(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "full_transcript": "dialogue",
                    "speakers": ["Alice"],
                    "consolidated_visual_captions": [],
                    "consolidated_actions": [],
                    "consolidated_audio_events": [],
                }
            ]
        )
    )

    def fake_ollama_client(_prompt, _config):
        return {
            "segment_id": "overwritten",
            "start_time": 999,
            "speakers": ["Mallory"],
            "title": "  Generated Title  ",
            "summary": "  Generated summary.  ",
            "keywords": [" safe ", "", 42],
        }

    run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": "ollama"},
        },
        llm_clients={"ollama": fake_ollama_client},
    )

    [segment] = json.loads((tmp_path / "enriched.json").read_text())
    assert segment["segment_id"] == "segment_0001"
    assert segment["start_time"] == 10.0
    assert segment["speakers"] == ["Alice"]
    assert segment["title"] == "Generated Title"
    assert segment["summary"] == "Generated summary."
    assert segment["keywords"] == ["safe", "42"]


def test_call_gemini_api_missing_key_does_not_require_google_sdk(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert _call_gemini_api(
        "prompt",
        {"llm_enrichment": {"gemini": {"model": "gemini-test"}}},
    ) is None


def test_call_gemini_api_blank_key_does_not_require_google_sdk(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", " ")
    imported_google_modules = []
    original_import = builtins.__import__

    def tracking_import(name, *args, **kwargs):
        if name.startswith("google"):
            imported_google_modules.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    assert _call_gemini_api(
        "prompt",
        {"llm_enrichment": {"gemini": {"model": "gemini-test"}}},
    ) is None
    assert imported_google_modules == []


def test_run_enrichment_rejects_unknown_provider_before_copying(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text("[]")

    result = run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": "unknown"},
        },
    )

    assert result is None
    assert not (tmp_path / "enriched.json").exists()


@pytest.mark.parametrize(
    "segments",
    [
        {"segment_id": "segment_0001"},
        ["not a segment"],
        [{"summary": "missing id"}],
        [{"segment_id": " "}],
        [{"segment_id": "segment_0001", "full_transcript": 42}],
        [{"segment_id": "segment_0001", "speakers": "Alice"}],
        [{"segment_id": "segment_0001", "consolidated_visual_captions": [42]}],
        [{"segment_id": "segment_0001", "consolidated_actions": [None]}],
        [{"segment_id": "segment_0001", "consolidated_audio_events": ["music", 7]}],
    ],
)
def test_run_enrichment_rejects_invalid_source_segments_before_copying_or_calling_provider(
    tmp_path,
    segments,
):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(json.dumps(segments))
    calls = []

    result = run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": "ollama"},
        },
        llm_clients={"ollama": lambda _prompt, _config: calls.append("called")},
    )

    assert result is None
    assert calls == []
    assert not (tmp_path / "enriched.json").exists()


def test_run_enrichment_rejects_invalid_resume_state_before_calling_provider(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(json.dumps([{"segment_id": "segment_0001"}]))
    (tmp_path / "enriched.json").write_text(
        json.dumps([{"segment_id": "segment_0001", "speakers": "Alice"}])
    )
    calls = []

    result = run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": "ollama"},
        },
        llm_clients={"ollama": lambda _prompt, _config: calls.append("called")},
    )

    assert result is None
    assert calls == []
