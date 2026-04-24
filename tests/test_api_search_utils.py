from api.search_utils import text_metadata_by_segment_id


def test_text_metadata_lookup_uses_exact_segment_ids():
    metadata = text_metadata_by_segment_id(
        ids=["10_text", "1_text"],
        metadatas=[{"title": "ten"}, {"title": "one"}],
    )

    assert metadata["1"] == {"title": "one"}
    assert metadata["10"] == {"title": "ten"}
