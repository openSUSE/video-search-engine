import pytest
from pydantic import ValidationError

from api.schemas import SearchQuery, SearchResult


def _search_result_payload(**updates):
    payload = {
        "id": "segment-a",
        "score": 0.5,
        "start_time": 1,
        "end_time": 2.5,
        "title": "Opening",
        "summary": "A calm opening scene.",
        "video_filename": "demo",
        "speakers": "Alice",
    }
    payload.update(updates)
    return payload


def test_search_query_strips_text_fields():
    query = SearchQuery(query="  find this  ", video_filename="  demo  ")

    assert query.query == "find this"
    assert query.video_filename == "demo"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "find this", "top_k": 0},
        {"query": "find this", "top_k": 51},
        {"query": "find this", "top_k": True},
        {"query": "find this", "top_k": "5"},
        {"query": "find this", "video_filename": "  "},
        {"query": "find this", "video_filename": "."},
        {"query": "find this", "video_filename": ".."},
        {"query": "find this", "video_filename": "nested/demo"},
        {"query": "find this", "video_filename": "nested\\demo"},
        {"query": "x" * 1001},
        {"query": "find this", "video_filename": "v" * 513 + ".mp4"},
    ],
)
def test_search_query_rejects_unusable_inputs(payload):
    with pytest.raises(ValidationError):
        SearchQuery(**payload)


def test_search_query_accepts_inputs_at_length_bounds():
    query = SearchQuery(
        query="x" * 1000,
        video_filename="v" * 512,
    )

    assert len(query.query) == 1000
    assert len(query.video_filename) == 512


def test_search_result_strips_display_strings_and_allows_empty_speakers():
    result = SearchResult(
        **_search_result_payload(
            id="  segment-a  ",
            title="  Opening  ",
            summary="  A calm opening scene.  ",
            video_filename="  demo  ",
            speakers="  ",
        )
    )

    assert result.id == "segment-a"
    assert result.title == "Opening"
    assert result.summary == "A calm opening scene."
    assert result.video_filename == "demo"
    assert result.speakers == ""


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"id": ""}, "id"),
        ({"id": "   "}, "id"),
        ({"title": ""}, "title"),
        ({"summary": "   "}, "summary"),
        ({"video_filename": ""}, "video_filename"),
        ({"video_filename": "."}, "video_filename"),
        ({"video_filename": ".."}, "video_filename"),
        ({"video_filename": "nested/demo"}, "video_filename"),
        ({"video_filename": "nested\\demo"}, "video_filename"),
        ({"score": -0.1}, "score"),
        ({"score": float("nan")}, "score"),
        ({"score": True}, "score"),
        ({"score": "0.5"}, "score"),
        ({"start_time": -1}, "start_time"),
        ({"start_time": True}, "start_time"),
        ({"start_time": "1"}, "start_time"),
        ({"end_time": float("inf")}, "end_time"),
        ({"end_time": "2"}, "end_time"),
        ({"start_time": 3, "end_time": 2}, "end_time"),
    ],
)
def test_search_result_rejects_unusable_output_fields(updates, message):
    with pytest.raises(ValidationError, match=message):
        SearchResult(**_search_result_payload(**updates))
