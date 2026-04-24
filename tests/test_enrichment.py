import json

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


def test_call_gemini_api_missing_key_does_not_require_google_sdk(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert _call_gemini_api(
        "prompt",
        {"llm_enrichment": {"gemini": {"model": "gemini-test"}}},
    ) is None


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
