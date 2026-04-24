import json

from ingestion_pipeline.steps.step_02_segmentation import run_segmentation


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": texts, "kwargs": kwargs})
        vectors = {
            "hello there": [1.0, 0.0],
            "still hello": [1.0, 0.0],
            "new topic": [0.0, 1.0],
            "quiet room": [1.0, 0.0],
            "fast car": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]


def test_run_segmentation_uses_injected_model_and_configured_output_name(tmp_path):
    analysis_path = tmp_path / "analysis.json"
    speaker_map_path = tmp_path / "speaker_map.json"
    analysis_path.write_text(
        json.dumps(
            [
                {
                    "shot_id": "shot_0001",
                    "time_start_sec": 0.0,
                    "time_end_sec": 1.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "text": "hello there",
                            "speaker": "SPEAKER_00",
                        }
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                },
                {
                    "shot_id": "shot_0002",
                    "time_start_sec": 1.0,
                    "time_end_sec": 2.0,
                    "visual_caption": "quiet room",
                    "transcript_segments": [
                        {
                            "start": 1.0,
                            "end": 2.0,
                            "text": "still hello",
                            "speaker": "SPEAKER_00",
                        }
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                },
                {
                    "shot_id": "shot_0003",
                    "time_start_sec": 2.0,
                    "time_end_sec": 3.0,
                    "visual_caption": "fast car",
                    "transcript_segments": [
                        {
                            "start": 2.0,
                            "end": 3.0,
                            "text": "new topic",
                            "speaker": "SPEAKER_01",
                        }
                    ],
                    "audio_events": [],
                    "detected_actions": [],
                },
            ]
        )
    )
    speaker_map_path.write_text(json.dumps({"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}))
    embedding_model = FakeEmbeddingModel()

    output_path = run_segmentation(
        video_path="unused.mp4",
        analysis_path=str(analysis_path),
        speaker_map_path=str(speaker_map_path),
        config={"filenames": {"final_segments": "custom_segments.json"}},
        embedding_model=embedding_model,
    )

    assert output_path == str(tmp_path / "custom_segments.json")
    segments = json.loads((tmp_path / "custom_segments.json").read_text())
    assert [segment["shot_ids"] for segment in segments] == [
        ["shot_0001", "shot_0002"],
        ["shot_0003"],
    ]
    assert segments[0]["full_transcript"] == "hello there still hello"
    assert segments[0]["speakers"] == ["Alice"]
    assert segments[1]["speakers"] == ["Bob"]
    assert embedding_model.calls == [
        {
            "texts": ["hello there", "still hello", "new topic"],
            "kwargs": {"show_progress_bar": True, "normalize_embeddings": True},
        },
        {
            "texts": ["quiet room", "quiet room", "fast car"],
            "kwargs": {"show_progress_bar": True, "normalize_embeddings": True},
        },
    ]
