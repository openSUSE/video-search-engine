"""Benchmarks for the Streamlit-side helpers.

These run on every search request the UI makes and on every video selection
inside the Streamlit session, so even small regressions add up. The functions
exercised here are pure (no Streamlit runtime required) and form the
contract between the UI and the FastAPI service.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.ui import search_client, search_state, speaker_support, url_settings

from .harness import Benchmark


class _FakeResponse:
    """Stand-in for ``requests.Response`` used by ``search_results_from_response``."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> Dict[str, Any]:
        return self._payload


def _setup_search_payload():
    return ("interesting query", "demo.mp4", 5)


def _bench_search_payload(args):
    query, video_filename, top_k = args
    search_client.search_payload(query, video_filename, top_k=top_k)


def _setup_search_results_response() -> _FakeResponse:
    payload = {
        "results": [
            {
                "id": f"demo.mp4::segment_{i:04d}",
                "score": 0.5 + i * 0.001,
                "start_time": float(i),
                "end_time": float(i) + 5.0,
                "title": f"Segment {i}",
                "summary": "A representative summary.",
                "video_filename": "demo.mp4",
                "speakers": "Alice,Bob",
            }
            for i in range(20)
        ]
    }
    return _FakeResponse(payload)


def _bench_search_results_response(response: _FakeResponse) -> None:
    search_client.search_results_from_response(response)


def _bench_search_api_url(_: Any) -> None:
    search_client.search_api_url(
        {"api_server": {"host": "127.0.0.1", "port": 1234}}
    )


def _bench_search_timeout(_: Any) -> None:
    search_client.search_timeout_seconds("12.5")


def _setup_session_state():
    return {}


def _bench_ensure_search_session(state: Dict[str, Any]) -> None:
    search_state.ensure_search_session_state(state)


def _bench_reset_for_video(state: Dict[str, Any]) -> None:
    state.clear()
    search_state.reset_search_session_for_video(
        state, "demo.mp4", "/data/videos/demo.mp4", "demo"
    )


def _setup_speaker_map():
    return {f"SPEAKER_{i:02d}": f"Person {i}" for i in range(20)}


def _bench_normalize_speaker_map(speaker_map: Dict[str, str]) -> None:
    speaker_support.normalize_speaker_map(speaker_map)


def _setup_transcript_segments() -> List[Dict[str, Any]]:
    return [
        {
            "speaker": f"SPEAKER_{i % 5:02d}",
            "text": f"Some dialogue {i}",
            "start": float(i),
            "end": float(i) + 1.0,
        }
        for i in range(200)
    ]


def _bench_transcript_validation(segments: List[Dict[str, Any]]) -> None:
    speaker_support.validate_transcript_segments_for_display(segments)


def _bench_speaker_ids_from_transcript(segments: List[Dict[str, Any]]) -> None:
    speaker_support.speaker_ids_from_transcript(segments)


def _bench_local_http_url(_: Any) -> None:
    url_settings.local_http_url("0.0.0.0", 1234)
    url_settings.local_http_url("127.0.0.1", "8501")
    url_settings.local_http_url("::1", "8000")


BENCHMARKS = [
    Benchmark(
        name="search_client.search_payload",
        category="ui",
        description="Build the JSON payload sent on every /search request.",
        fn=_bench_search_payload,
        setup=_setup_search_payload,
        iterations=50_000,
    ),
    Benchmark(
        name="search_client.search_results_from_response(20)",
        category="ui",
        description=(
            "Validate and normalise a 20-item /search response coming from "
            "the FastAPI service."
        ),
        fn=_bench_search_results_response,
        setup=_setup_search_results_response,
        iterations=2_000,
        inner_loops=20,
    ),
    Benchmark(
        name="search_client.search_api_url",
        category="ui",
        description="Construct the /search endpoint URL from a config mapping.",
        fn=_bench_search_api_url,
        iterations=50_000,
    ),
    Benchmark(
        name="search_client.search_timeout_seconds",
        category="ui",
        description="Parse SEARCH_API_TIMEOUT_SECONDS-style values.",
        fn=_bench_search_timeout,
        iterations=50_000,
    ),
    Benchmark(
        name="search_state.ensure_search_session_state",
        category="ui",
        description="Initialise the empty Streamlit search session dict.",
        fn=_bench_ensure_search_session,
        setup=_setup_session_state,
        iterations=50_000,
    ),
    Benchmark(
        name="search_state.reset_search_session_for_video",
        category="ui",
        description="Switch the active video and clear search results.",
        fn=_bench_reset_for_video,
        setup=_setup_session_state,
        iterations=50_000,
    ),
    Benchmark(
        name="speaker_support.normalize_speaker_map",
        category="ui",
        description="Validate and normalise a 20-entry speaker map.",
        fn=_bench_normalize_speaker_map,
        setup=_setup_speaker_map,
        iterations=20_000,
    ),
    Benchmark(
        name="speaker_support.validate_transcript_segments(200)",
        category="ui",
        description="Validate a 200-segment transcript for the speaker UI.",
        fn=_bench_transcript_validation,
        setup=_setup_transcript_segments,
        iterations=1_000,
    ),
    Benchmark(
        name="speaker_support.speaker_ids_from_transcript(200)",
        category="ui",
        description="Extract distinct speaker ids from a 200-segment transcript.",
        fn=_bench_speaker_ids_from_transcript,
        setup=_setup_transcript_segments,
        iterations=2_000,
    ),
    Benchmark(
        name="url_settings.local_http_url.x3",
        category="ui",
        description="Three URL constructions covering wildcard/loopback/IPv6.",
        fn=_bench_local_http_url,
        iterations=50_000,
        inner_loops=3,
    ),
]
