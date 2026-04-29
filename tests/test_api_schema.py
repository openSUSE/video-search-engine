import pytest
from pydantic import ValidationError

from api.schemas import SearchQuery


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
        {"query": "find this", "video_filename": "  "},
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
