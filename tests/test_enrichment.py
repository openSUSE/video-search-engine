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
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "full_transcript": "important dialogue",
                    "speakers": ["Alice"],
                    "consolidated_visual_captions": ["a train platform"],
                    "consolidated_actions": ["waiting"],
                    "consolidated_audio_events": ["announcement"],
                },
                {
                    "segment_id": "segment_0002",
                    "start_time": 5.0,
                    "end_time": 10.0,
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


def test_run_enrichment_normalizes_context_lists_before_prompting(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "full_transcript": "dialogue",
                    "speakers": [" Alice ", "", "  Bob  "],
                    "consolidated_visual_captions": [" platform ", "   "],
                    "consolidated_actions": [" waiting ", ""],
                    "consolidated_audio_events": [" announcement ", "\t"],
                }
            ]
        )
    )
    calls = []

    def fake_ollama_client(prompt, _config):
        calls.append(prompt)
        return {
            "title": "Generated Title",
            "summary": "Generated summary.",
            "keywords": ["platform"],
        }

    output_path = run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": "ollama"},
        },
        llm_clients={"ollama": fake_ollama_client},
    )

    assert output_path == str(tmp_path / "enriched.json")
    assert "- Speakers: Alice, Bob" in calls[0]
    assert "- Key Visuals: platform" in calls[0]
    assert "- Key Actions: waiting" in calls[0]
    assert "- Background Audio Events: announcement" in calls[0]
    enriched_segments = json.loads((tmp_path / "enriched.json").read_text())
    assert enriched_segments[0]["speakers"] == ["Alice", "Bob"]
    assert enriched_segments[0]["consolidated_visual_captions"] == ["platform"]
    assert enriched_segments[0]["consolidated_actions"] == ["waiting"]
    assert enriched_segments[0]["consolidated_audio_events"] == ["announcement"]


def test_run_enrichment_normalizes_provider_name(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.0,
                    "end_time": 5.0,
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
                    "start_time": 0.0,
                    "end_time": 5.0,
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
                    "start_time": 0.0,
                    "end_time": 5.0,
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


def test_run_enrichment_ignores_unreadable_video_metadata(
    monkeypatch,
    tmp_path,
):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "full_transcript": "dialogue",
                    "speakers": [],
                    "consolidated_visual_captions": [],
                    "consolidated_actions": [],
                    "consolidated_audio_events": [],
                }
            ]
        )
    )
    metadata_path = tmp_path / "video_metadata.json"
    metadata_path.write_text(json.dumps({"title": "Demo Movie"}))
    real_open = builtins.open

    def unreadable_metadata(path, *args, **kwargs):
        if path == str(metadata_path):
            raise PermissionError("permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", unreadable_metadata)
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
                    "start_time": 0.0,
                    "end_time": 5.0,
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


@pytest.mark.parametrize(
    "resume_updates",
    [
        {"title": {"bad": "shape"}},
        {"summary": ["bad", "shape"]},
        {"keywords": ["existing", 42]},
    ],
)
def test_run_enrichment_retries_malformed_enrichment_fields(tmp_path, resume_updates):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "full_transcript": "dialogue",
                }
            ]
        )
    )
    resume_segment = {
        "segment_id": "segment_0001",
        "start_time": 0.0,
        "end_time": 5.0,
        "full_transcript": "dialogue",
        "title": "Existing Title",
        "summary": "Existing summary.",
        "keywords": ["existing"],
    }
    resume_segment.update(resume_updates)
    (tmp_path / "enriched.json").write_text(json.dumps([resume_segment]))
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


def test_run_enrichment_reinitializes_resume_when_segment_ids_change(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "full_transcript": "current dialogue",
                }
            ]
        )
    )
    (tmp_path / "enriched.json").write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_9999",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "full_transcript": "stale dialogue",
                    "title": "Stale Title",
                    "summary": "Stale summary.",
                    "keywords": ["stale"],
                }
            ]
        )
    )
    calls = []

    def fake_ollama_client(prompt, _config):
        calls.append(prompt)
        return {
            "title": "Fresh Title",
            "summary": "Fresh summary.",
            "keywords": ["fresh"],
        }

    output_path = run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": "ollama"},
        },
        llm_clients={"ollama": fake_ollama_client},
    )

    assert output_path == str(tmp_path / "enriched.json")
    [segment] = json.loads((tmp_path / "enriched.json").read_text())
    assert len(calls) == 1
    assert "current dialogue" in calls[0]
    assert segment["segment_id"] == "segment_0001"
    assert segment["full_transcript"] == "current dialogue"
    assert segment["title"] == "Fresh Title"


def test_run_enrichment_reinitializes_resume_when_source_context_changes(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "full_transcript": "current dialogue",
                }
            ]
        )
    )
    (tmp_path / "enriched.json").write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "full_transcript": "stale dialogue",
                    "title": "Stale Title",
                    "summary": "Stale summary.",
                    "keywords": ["stale"],
                }
            ]
        )
    )
    calls = []

    def fake_ollama_client(prompt, _config):
        calls.append(prompt)
        return {
            "title": "Fresh Title",
            "summary": "Fresh summary.",
            "keywords": ["fresh"],
        }

    output_path = run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": "ollama"},
        },
        llm_clients={"ollama": fake_ollama_client},
    )

    assert output_path == str(tmp_path / "enriched.json")
    [segment] = json.loads((tmp_path / "enriched.json").read_text())
    assert len(calls) == 1
    assert "current dialogue" in calls[0]
    assert "stale dialogue" not in calls[0]
    assert segment["full_transcript"] == "current dialogue"
    assert segment["title"] == "Fresh Title"


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
    assert segment["keywords"] == ["safe"]


def test_run_enrichment_splits_delimited_llm_keyword_strings(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "full_transcript": "dialogue",
                    "speakers": ["Alice"],
                    "consolidated_visual_captions": ["station platform"],
                    "consolidated_actions": ["waiting"],
                    "consolidated_audio_events": ["announcement"],
                }
            ]
        )
    )

    def fake_ollama_client(_prompt, _config):
        return {
            "title": "Generated Title",
            "summary": "Generated summary.",
            "keywords": "platform, station\nannouncement; crowd",
        }

    output_path = run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": "ollama"},
        },
        llm_clients={"ollama": fake_ollama_client},
    )

    assert output_path == str(tmp_path / "enriched.json")
    [segment] = json.loads((tmp_path / "enriched.json").read_text())
    assert segment["keywords"] == ["platform", "station", "announcement", "crowd"]


def test_run_enrichment_parses_json_encoded_llm_keyword_strings(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "full_transcript": "dialogue",
                    "speakers": ["Alice"],
                    "consolidated_visual_captions": ["station platform"],
                    "consolidated_actions": ["waiting"],
                    "consolidated_audio_events": ["announcement"],
                }
            ]
        )
    )

    def fake_ollama_client(_prompt, _config):
        return {
            "title": "Generated Title",
            "summary": "Generated summary.",
            "keywords": '["platform", "station", "announcement"]',
        }

    output_path = run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": "ollama"},
        },
        llm_clients={"ollama": fake_ollama_client},
    )

    assert output_path == str(tmp_path / "enriched.json")
    [segment] = json.loads((tmp_path / "enriched.json").read_text())
    assert segment["keywords"] == ["platform", "station", "announcement"]


def test_run_enrichment_does_not_stringify_malformed_llm_text_fields(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "full_transcript": "dialogue",
                }
            ]
        )
    )

    def malformed_ollama_client(_prompt, _config):
        return {
            "title": {"bad": "shape"},
            "summary": ["bad", "shape"],
            "keywords": ["safe", {"bad": "shape"}, 7, " useful "],
        }

    result = run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": "ollama"},
        },
        llm_clients={"ollama": malformed_ollama_client},
    )

    assert result is None
    [segment] = json.loads((tmp_path / "enriched.json").read_text())
    assert "title" not in segment
    assert "summary" not in segment
    assert segment["keywords"] == ["safe", "useful"]


def test_run_enrichment_returns_none_when_any_segment_remains_incomplete(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "full_transcript": "first dialogue",
                },
                {
                    "segment_id": "segment_0002",
                    "start_time": 5.0,
                    "end_time": 10.0,
                    "full_transcript": "second dialogue",
                },
            ]
        )
    )
    calls = []

    def flaky_ollama_client(_prompt, _config):
        calls.append(len(calls))
        if len(calls) == 1:
            return {
                "title": "Generated Title",
                "summary": "Generated summary.",
                "keywords": ["complete"],
            }
        return None

    result = run_enrichment(
        str(segments_path),
        {
            "filenames": {"enriched_segments": "enriched.json"},
            "llm_enrichment": {"provider": "ollama"},
        },
        llm_clients={"ollama": flaky_ollama_client},
    )

    assert result is None
    enriched_segments = json.loads((tmp_path / "enriched.json").read_text())
    assert enriched_segments[0]["title"] == "Generated Title"
    assert enriched_segments[0]["keywords"] == ["complete"]
    assert enriched_segments[1]["title"] == "Error"
    assert enriched_segments[1]["summary"] == "Failed to generate"
    assert enriched_segments[1]["keywords"] == []


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


def test_run_enrichment_rejects_empty_source_segments_before_copying_or_calling_provider(
    tmp_path,
):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text("[]")
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


@pytest.mark.parametrize(
    "segments",
    [
        {"segment_id": "segment_0001"},
        ["not a segment"],
        [{"summary": "missing id"}],
        [{"segment_id": " "}],
        [{"segment_id": "chapter::segment_0001", "start_time": 0.0, "end_time": 1.0}],
        [{"segment_id": "segment_0001", "full_transcript": 42}],
        [{"segment_id": "segment_0001", "speakers": "Alice"}],
        [{"segment_id": "segment_0001", "consolidated_visual_captions": [42]}],
        [{"segment_id": "segment_0001", "consolidated_actions": [None]}],
        [{"segment_id": "segment_0001", "consolidated_audio_events": ["music", 7]}],
        [{"segment_id": "segment_0001", "end_time": 1.0}],
        [{"segment_id": "segment_0001", "start_time": 0.0}],
        [{"segment_id": "segment_0001", "start_time": "0", "end_time": 1.0}],
        [{"segment_id": "segment_0001", "start_time": 0.0, "end_time": "1"}],
        [{"segment_id": "segment_0001", "start_time": True, "end_time": 1.0}],
        [{"segment_id": "segment_0001", "start_time": float("nan"), "end_time": 1.0}],
        [{"segment_id": "segment_0001", "start_time": 2.0, "end_time": 1.0}],
        [{"segment_id": "segment_0001", "start_time": 0.0, "end_time": float("inf")}],
        [
            {"segment_id": "segment_0001", "start_time": 0.0, "end_time": 1.0},
            {"segment_id": " segment_0001 ", "start_time": 1.0, "end_time": 2.0},
        ],
        [
            {"segment_id": "segment_0001", "start_time": 5.0, "end_time": 6.0},
            {"segment_id": "segment_0002", "start_time": 4.0, "end_time": 5.0},
        ],
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


def test_run_enrichment_rejects_duplicate_resume_segment_ids_before_calling_provider(
    tmp_path,
):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.0,
                    "end_time": 5.0,
                }
            ]
        )
    )
    (tmp_path / "enriched.json").write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.0,
                    "end_time": 5.0,
                },
                {
                    "segment_id": " segment_0001 ",
                    "start_time": 5.0,
                    "end_time": 10.0,
                },
            ]
        )
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


def test_run_enrichment_rejects_invalid_resume_state_before_calling_provider(tmp_path):
    segments_path = tmp_path / "final_segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.0,
                    "end_time": 5.0,
                }
            ]
        )
    )
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
