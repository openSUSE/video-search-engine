import logging

import pytest

from api.search_service import HybridSearchService, RRF_K, _first_id_list


def _metadata(**updates):
    metadata = {
        "title": "Title",
        "summary": "Summary",
        "start_time": 1.0,
        "end_time": 2.0,
        "video_filename": "demo.mp4",
        "speakers": "Alice",
    }
    metadata.update(updates)
    return metadata


class FakeEmbeddingModel:
    def __init__(self):
        self.queries = []

    def encode(self, query):
        self.queries.append(query)
        return [0.25, 0.75]


class FakeCollection:
    def __init__(self):
        self.query_calls = []
        self.get_call = None

    def query(self, *, query_embeddings, n_results, where, include):
        self.query_calls.append(
            {
                "query_embeddings": query_embeddings,
                "n_results": n_results,
                "where": where,
                "include": include,
            }
        )
        doc_type = where["type"] if "type" in where else where["$and"][0]["type"]
        if doc_type == "text":
            return {"ids": [["demo.mp4::segment-a_text", "demo.mp4::segment-b_text"]]}
        return {"ids": [["demo.mp4::segment-b_visual", "demo.mp4::segment-c_visual"]]}

    def get(self, *, ids, include):
        self.get_call = {"ids": ids, "include": include}
        return {
            "ids": ["demo.mp4::segment-b_text", "demo.mp4::segment-a_text"],
            "metadatas": [
                _metadata(title="B", summary="shared text and visual match"),
                _metadata(title="A", summary="text-only match"),
            ],
        }


def test_hybrid_search_service_reranks_and_fetches_text_metadata():
    model = FakeEmbeddingModel()
    collection = FakeCollection()
    service = HybridSearchService(model, collection)

    results = service.search("find highlights", top_k=2, video_filename="demo.mp4")

    assert model.queries == ["find highlights"]
    assert [result["id"] for result in results] == [
        "demo.mp4::segment-b",
        "demo.mp4::segment-a",
    ]
    assert results[0]["title"] == "B"
    assert results[1]["summary"] == "text-only match"

    assert collection.query_calls == [
        {
            "query_embeddings": [[0.25, 0.75]],
            "n_results": 6,
            "where": {"$and": [{"type": "text"}, {"video_filename": "demo.mp4"}]},
            "include": [],
        },
        {
            "query_embeddings": [[0.25, 0.75]],
            "n_results": 6,
            "where": {"$and": [{"type": "visual"}, {"video_filename": "demo.mp4"}]},
            "include": [],
        },
    ]
    assert collection.get_call == {
        "ids": [
            "demo.mp4::segment-b_text",
            "demo.mp4::segment-a_text",
            "demo.mp4::segment-c_text",
        ],
        "include": ["metadatas"],
    }


def test_hybrid_search_service_strips_query_and_video_filter_before_search():
    model = FakeEmbeddingModel()
    collection = FakeCollection()
    service = HybridSearchService(model, collection)

    service.search("  find highlights  ", top_k=1, video_filename="  demo.mp4  ")

    assert model.queries == ["find highlights"]
    assert [call["where"] for call in collection.query_calls] == [
        {"$and": [{"type": "text"}, {"video_filename": "demo.mp4"}]},
        {"$and": [{"type": "visual"}, {"video_filename": "demo.mp4"}]},
    ]


def test_hybrid_search_service_skips_malformed_text_metadata():
    class CollectionWithStaleMetadata(FakeCollection):
        def get(self, *, ids, include):
            self.get_call = {"ids": ids, "include": include}
            return {
                "ids": ["demo.mp4::segment-b_text", "demo.mp4::segment-a_text"],
                "metadatas": [
                    _metadata(title="B", summary="valid result"),
                    _metadata(title="A", start_time=float("nan")),
                ],
            }

    service = HybridSearchService(FakeEmbeddingModel(), CollectionWithStaleMetadata())

    results = service.search("find highlights", top_k=2, video_filename="demo.mp4")

    assert [result["id"] for result in results] == ["demo.mp4::segment-b"]
    assert results[0]["summary"] == "valid result"


def test_hybrid_search_service_skips_metadata_for_wrong_video_filter():
    class CollectionWithWrongVideoMetadata(FakeCollection):
        def query(self, *, query_embeddings, n_results, where, include):
            self.query_calls.append(
                {
                    "query_embeddings": query_embeddings,
                    "n_results": n_results,
                    "where": where,
                    "include": include,
                }
            )
            doc_type = where["type"] if "type" in where else where["$and"][0]["type"]
            if doc_type == "text":
                return {
                    "ids": [
                        [
                            "demo.mp4::wrong-video_text",
                            "demo.mp4::valid_text",
                        ]
                    ]
                }
            return {"ids": [[]]}

        def get(self, *, ids, include):
            self.get_call = {"ids": ids, "include": include}
            return {
                "ids": [
                    "demo.mp4::wrong-video_text",
                    "demo.mp4::valid_text",
                ],
                "metadatas": [
                    _metadata(title="Wrong video", video_filename="other.mp4"),
                    _metadata(title="Valid", summary="matching selected video"),
                ],
            }

    service = HybridSearchService(
        FakeEmbeddingModel(),
        CollectionWithWrongVideoMetadata(),
    )

    results = service.search("find highlights", top_k=2, video_filename="demo.mp4")

    assert [result["id"] for result in results] == ["demo.mp4::valid"]
    assert results[0]["summary"] == "matching selected video"


def test_hybrid_search_service_skips_metadata_mismatching_scoped_document_id():
    class CollectionWithMismatchedDocumentScope(FakeCollection):
        def query(self, *, query_embeddings, n_results, where, include):
            self.query_calls.append(
                {
                    "query_embeddings": query_embeddings,
                    "n_results": n_results,
                    "where": where,
                    "include": include,
                }
            )
            doc_type = where["type"] if "type" in where else where["$and"][0]["type"]
            if doc_type == "text":
                return {
                    "ids": [
                        [
                            "demo.mp4::wrong-scope_text",
                            "other.mp4::valid_text",
                        ]
                    ]
                }
            return {"ids": [[]]}

        def get(self, *, ids, include):
            self.get_call = {"ids": ids, "include": include}
            return {
                "ids": [
                    "demo.mp4::wrong-scope_text",
                    "other.mp4::valid_text",
                ],
                "metadatas": [
                    _metadata(title="Wrong scope", video_filename="other.mp4"),
                    _metadata(
                        title="Other video",
                        summary="metadata matches document scope",
                        video_filename="other.mp4",
                    ),
                ],
            }

    service = HybridSearchService(
        FakeEmbeddingModel(),
        CollectionWithMismatchedDocumentScope(),
    )

    results = service.search("find highlights", top_k=2)

    assert [result["id"] for result in results] == ["other.mp4::valid"]
    assert results[0]["summary"] == "metadata matches document scope"


def test_hybrid_search_service_backfills_from_lower_ranked_candidates_when_top_metadata_stale():
    class CollectionWithStaleTopCandidate(FakeCollection):
        def query(self, *, query_embeddings, n_results, where, include):
            self.query_calls.append(
                {
                    "query_embeddings": query_embeddings,
                    "n_results": n_results,
                    "where": where,
                    "include": include,
                }
            )
            doc_type = where["type"] if "type" in where else where["$and"][0]["type"]
            if doc_type == "text":
                return {
                    "ids": [
                        [
                            "demo.mp4::stale_text",
                            "demo.mp4::primary_text",
                            "demo.mp4::fallback_text",
                        ]
                    ]
                }
            return {"ids": [[]]}

        def get(self, *, ids, include):
            self.get_call = {"ids": ids, "include": include}
            return {
                "ids": [
                    "demo.mp4::stale_text",
                    "demo.mp4::primary_text",
                    "demo.mp4::fallback_text",
                ],
                "metadatas": [
                    _metadata(title=" "),
                    _metadata(title="Primary", summary="top valid candidate"),
                    _metadata(title="Fallback", summary="lower valid candidate"),
                ],
            }

    collection = CollectionWithStaleTopCandidate()
    service = HybridSearchService(FakeEmbeddingModel(), collection)

    results = service.search("find highlights", top_k=2)

    assert collection.get_call == {
        "ids": [
            "demo.mp4::stale_text",
            "demo.mp4::primary_text",
            "demo.mp4::fallback_text",
        ],
        "include": ["metadatas"],
    }
    assert [result["id"] for result in results] == [
        "demo.mp4::primary",
        "demo.mp4::fallback",
    ]


def test_hybrid_search_service_strips_whitespace_from_returned_metadata():
    class CollectionWithPaddedFields(FakeCollection):
        def get(self, *, ids, include):
            self.get_call = {"ids": ids, "include": include}
            return {
                "ids": ["demo.mp4::segment-b_text", "demo.mp4::segment-a_text"],
                "metadatas": [
                    _metadata(
                        title="  Padded Title  ",
                        summary=" trimmed summary ",
                        speakers=" Alice, Bob ",
                    ),
                    _metadata(title="A", summary="text-only match"),
                ],
            }

    service = HybridSearchService(FakeEmbeddingModel(), CollectionWithPaddedFields())

    results = service.search("find highlights", top_k=2, video_filename="demo.mp4")

    by_id = {result["id"]: result for result in results}
    assert by_id["demo.mp4::segment-b"]["title"] == "Padded Title"
    assert by_id["demo.mp4::segment-b"]["summary"] == "trimmed summary"
    assert by_id["demo.mp4::segment-b"]["speakers"] == "Alice, Bob"


@pytest.mark.parametrize(
    "blank_field",
    [
        {"title": ""},
        {"title": "   "},
        {"summary": ""},
        {"summary": "   "},
    ],
)
def test_hybrid_search_service_skips_results_with_blank_title_or_summary(blank_field):
    class CollectionWithBlankField(FakeCollection):
        def get(self, *, ids, include):
            self.get_call = {"ids": ids, "include": include}
            return {
                "ids": ["demo.mp4::segment-b_text", "demo.mp4::segment-a_text"],
                "metadatas": [
                    _metadata(title="B", summary="kept result"),
                    _metadata(**blank_field),
                ],
            }

    service = HybridSearchService(FakeEmbeddingModel(), CollectionWithBlankField())

    results = service.search("find highlights", top_k=2, video_filename="demo.mp4")

    assert [result["id"] for result in results] == ["demo.mp4::segment-b"]


def test_hybrid_search_service_skips_stringified_time_metadata():
    class CollectionWithStringifiedTimes(FakeCollection):
        def query(self, *, query_embeddings, n_results, where, include):
            self.query_calls.append(
                {
                    "query_embeddings": query_embeddings,
                    "n_results": n_results,
                    "where": where,
                    "include": include,
                }
            )
            doc_type = where["type"] if "type" in where else where["$and"][0]["type"]
            if doc_type == "text":
                return {
                    "ids": [
                        [
                            "demo.mp4::string-time_text",
                            "demo.mp4::primary_text",
                            "demo.mp4::fallback_text",
                        ]
                    ]
                }
            return {"ids": [[]]}

        def get(self, *, ids, include):
            self.get_call = {"ids": ids, "include": include}
            return {
                "ids": [
                    "demo.mp4::string-time_text",
                    "demo.mp4::primary_text",
                    "demo.mp4::fallback_text",
                ],
                "metadatas": [
                    _metadata(title="String time", start_time="1", end_time="2"),
                    _metadata(title="Primary", summary="top valid candidate"),
                    _metadata(title="Fallback", summary="lower valid candidate"),
                ],
            }

    service = HybridSearchService(FakeEmbeddingModel(), CollectionWithStringifiedTimes())

    results = service.search("find highlights", top_k=2)

    assert [result["id"] for result in results] == [
        "demo.mp4::primary",
        "demo.mp4::fallback",
    ]


def test_hybrid_search_service_skips_malformed_fetched_metadata_ids():
    class CollectionWithMalformedFetchedMetadataIds(FakeCollection):
        def get(self, *, ids, include):
            self.get_call = {"ids": ids, "include": include}
            return {
                "ids": [
                    "demo.mp4::segment-b_text",
                    "demo.mp4::segment-a_visual",
                    123,
                    None,
                    "demo.mp4::segment-a_text",
                ],
                "metadatas": [
                    _metadata(title="B", summary="valid fetched result"),
                    _metadata(title="A visual", summary="wrong document type"),
                    _metadata(title="numeric id"),
                    _metadata(title="missing id"),
                    "not metadata",
                ],
            }

    service = HybridSearchService(
        FakeEmbeddingModel(),
        CollectionWithMalformedFetchedMetadataIds(),
    )

    results = service.search("find highlights", top_k=2, video_filename="demo.mp4")

    assert [result["id"] for result in results] == ["demo.mp4::segment-b"]
    assert results[0]["summary"] == "valid fetched result"


def test_hybrid_search_service_skips_malformed_query_result_ids_before_fusion():
    class CollectionWithMalformedQueryIds(FakeCollection):
        def query(self, *, query_embeddings, n_results, where, include):
            self.query_calls.append(
                {
                    "query_embeddings": query_embeddings,
                    "n_results": n_results,
                    "where": where,
                    "include": include,
                }
            )
            doc_type = where["type"] if "type" in where else where["$and"][0]["type"]
            if doc_type == "text":
                return {
                    "ids": [
                        [
                            "demo.mp4::segment-a_text",
                            "demo.mp4::wrong-visual-id_visual",
                            123,
                        ]
                    ]
                }
            return {
                "ids": [
                    [
                        "demo.mp4::segment-b_visual",
                        "demo.mp4::wrong-text-id_text",
                        None,
                    ]
                ]
            }

        def get(self, *, ids, include):
            self.get_call = {"ids": ids, "include": include}
            return {
                "ids": ["demo.mp4::segment-a_text", "demo.mp4::segment-b_text"],
                "metadatas": [
                    _metadata(title="A", summary="valid text result"),
                    _metadata(title="B", summary="valid visual result"),
                ],
            }

    collection = CollectionWithMalformedQueryIds()
    service = HybridSearchService(FakeEmbeddingModel(), collection)

    results = service.search("find highlights", top_k=3, video_filename="demo.mp4")

    assert [result["id"] for result in results] == [
        "demo.mp4::segment-a",
        "demo.mp4::segment-b",
    ]
    assert collection.get_call == {
        "ids": ["demo.mp4::segment-a_text", "demo.mp4::segment-b_text"],
        "include": ["metadatas"],
    }


def test_hybrid_search_service_skips_blank_segment_ids_before_fusion():
    class CollectionWithBlankSegmentIds(FakeCollection):
        def query(self, *, query_embeddings, n_results, where, include):
            self.query_calls.append(
                {
                    "query_embeddings": query_embeddings,
                    "n_results": n_results,
                    "where": where,
                    "include": include,
                }
            )
            doc_type = where["type"] if "type" in where else where["$and"][0]["type"]
            if doc_type == "text":
                return {
                    "ids": [
                        [
                            "_text",
                            "   _text",
                            "demo.mp4::valid_text",
                        ]
                    ]
                }
            return {"ids": [["_visual"]]}

        def get(self, *, ids, include):
            self.get_call = {"ids": ids, "include": include}
            return {
                "ids": ids,
                "metadatas": [
                    _metadata(title=f"Result {index}")
                    for index, _ in enumerate(ids)
                ],
            }

    collection = CollectionWithBlankSegmentIds()
    service = HybridSearchService(FakeEmbeddingModel(), collection)

    results = service.search("find highlights", top_k=3)

    assert [result["id"] for result in results] == ["demo.mp4::valid"]
    assert collection.get_call == {
        "ids": ["demo.mp4::valid_text"],
        "include": ["metadatas"],
    }


def test_hybrid_search_service_deduplicates_query_ids_before_fusion():
    class CollectionWithDuplicateQueryIds(FakeCollection):
        def query(self, *, query_embeddings, n_results, where, include):
            self.query_calls.append(
                {
                    "query_embeddings": query_embeddings,
                    "n_results": n_results,
                    "where": where,
                    "include": include,
                }
            )
            doc_type = where["type"] if "type" in where else where["$and"][0]["type"]
            if doc_type == "text":
                return {
                    "ids": [
                        [
                            "demo.mp4::duplicate_text",
                            "demo.mp4::duplicate_text",
                            "demo.mp4::text-only_text",
                        ]
                    ]
                }
            return {
                "ids": [
                    [
                        "demo.mp4::visual-only_visual",
                        "demo.mp4::visual-only_visual",
                    ]
                ]
            }

        def get(self, *, ids, include):
            self.get_call = {"ids": ids, "include": include}
            return {
                "ids": ids,
                "metadatas": [
                    _metadata(title="Duplicate"),
                    _metadata(title="Visual only"),
                    _metadata(title="Text only"),
                ],
            }

    collection = CollectionWithDuplicateQueryIds()
    service = HybridSearchService(FakeEmbeddingModel(), collection)

    results = service.search("find highlights", top_k=3)

    assert [result["id"] for result in results] == [
        "demo.mp4::duplicate",
        "demo.mp4::visual-only",
        "demo.mp4::text-only",
    ]
    assert results[0]["score"] == pytest.approx(1 / RRF_K)
    assert results[1]["score"] == pytest.approx(1 / RRF_K)
    assert results[2]["score"] == pytest.approx(1 / (RRF_K + 1))
    assert collection.get_call == {
        "ids": [
            "demo.mp4::duplicate_text",
            "demo.mp4::visual-only_text",
            "demo.mp4::text-only_text",
        ],
        "include": ["metadatas"],
    }


def test_hybrid_search_service_omits_video_filter_when_not_requested():
    collection = FakeCollection()
    service = HybridSearchService(FakeEmbeddingModel(), collection)

    service.search("find highlights", top_k=1)

    assert [call["where"] for call in collection.query_calls] == [
        {"type": "text"},
        {"type": "visual"},
    ]


@pytest.mark.parametrize("top_k", [0, -1, 51, True, 1.5, "5"])
def test_hybrid_search_service_rejects_invalid_top_k_before_embedding(top_k):
    model = FakeEmbeddingModel()
    collection = FakeCollection()
    service = HybridSearchService(model, collection)

    with pytest.raises(ValueError, match="top_k"):
        service.search("find highlights", top_k=top_k)

    assert model.queries == []
    assert collection.query_calls == []
    assert collection.get_call is None


@pytest.mark.parametrize("query", [None, "", "   ", 123, "x" * 1001])
def test_hybrid_search_service_rejects_invalid_query_before_embedding(query):
    model = FakeEmbeddingModel()
    collection = FakeCollection()
    service = HybridSearchService(model, collection)

    with pytest.raises(ValueError, match="query"):
        service.search(query, top_k=1)

    assert model.queries == []
    assert collection.query_calls == []
    assert collection.get_call is None


@pytest.mark.parametrize("video_filename", ["", "   ", 123, "v" * 513])
def test_hybrid_search_service_rejects_invalid_video_filter_before_embedding(
    video_filename,
):
    model = FakeEmbeddingModel()
    collection = FakeCollection()
    service = HybridSearchService(model, collection)

    with pytest.raises(ValueError, match="video_filename"):
        service.search("find highlights", top_k=1, video_filename=video_filename)

    assert model.queries == []
    assert collection.query_calls == []
    assert collection.get_call is None


@pytest.mark.parametrize(
    "encoded_query",
    [
        [],
        [0.25, "bad"],
        [0.25, float("inf")],
        [0.25, float("nan")],
        [[0.25, 0.75]],
    ],
)
def test_hybrid_search_service_rejects_invalid_query_embeddings_before_query(
    encoded_query,
):
    class BadEmbeddingModel:
        def encode(self, query):
            return encoded_query

    collection = FakeCollection()
    service = HybridSearchService(BadEmbeddingModel(), collection)

    with pytest.raises(ValueError, match="query embedding"):
        service.search("find highlights", top_k=1)

    assert collection.query_calls == []
    assert collection.get_call is None


@pytest.mark.parametrize(
    ("payload", "warning_substring"),
    [
        ("not a dict", "expected a mapping"),
        ({"ids": "not a list"}, "ids must be a list"),
        ({"ids": [None]}, "first ids row"),
        ({"ids": ["not a list"]}, "first ids row"),
        ({"ids": [42]}, "first ids row"),
    ],
)
def test_first_id_list_warns_on_malformed_shapes(caplog, payload, warning_substring):
    # ChromaDB's contract: results is a dict, results["ids"] is a list,
    # results["ids"][0] is a list/tuple of doc ids. Each rejection path
    # logs a distinct WARNING so an upstream regression is visible to
    # operators tailing the API logs.
    with caplog.at_level(logging.WARNING, logger="api.search_service"):
        result = _first_id_list(payload)

    assert result == []
    assert any(warning_substring in record.message for record in caplog.records)


def test_first_id_list_returns_empty_quietly_for_legitimate_empty_results(caplog):
    # An empty results.ids or first row of [] is a normal "no matches"
    # response from ChromaDB, not a malformed one — no warning should
    # fire, and we should just return an empty list to the caller.
    with caplog.at_level(logging.WARNING, logger="api.search_service"):
        assert _first_id_list({"ids": []}) == []
        assert _first_id_list({"ids": [[]]}) == []
        assert _first_id_list({}) == []

    assert not caplog.records
