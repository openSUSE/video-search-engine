from api.search_utils import text_metadata_by_segment_id


def test_text_metadata_lookup_uses_exact_segment_ids():
    metadata = text_metadata_by_segment_id(
        ids=["10_text", "1_text"],
        metadatas=[{"title": "ten"}, {"title": "one"}],
    )

    assert metadata["1"] == {"title": "one"}
    assert metadata["10"] == {"title": "ten"}


def test_text_metadata_lookup_skips_malformed_fetched_rows():
    metadata = text_metadata_by_segment_id(
        ids=[
            "valid_text",
            "visual-only_visual",
            123,
            None,
            "invalid-metadata_text",
        ],
        metadatas=[
            {"title": "valid"},
            {"title": "visual"},
            {"title": "numeric id"},
            {"title": "missing id"},
            "not metadata",
        ],
    )

    assert metadata == {"valid": {"title": "valid"}}
