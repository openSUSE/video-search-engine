from api.search_service import HybridSearchService


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

    def query(self, *, query_embeddings, n_results, where):
        self.query_calls.append(
            {
                "query_embeddings": query_embeddings,
                "n_results": n_results,
                "where": where,
            }
        )
        doc_type = where["$and"][0]["type"]
        if doc_type == "text":
            return {"ids": [["demo.mp4::segment-a_text", "demo.mp4::segment-b_text"]]}
        return {"ids": [["demo.mp4::segment-b_visual", "demo.mp4::segment-c_visual"]]}

    def get(self, *, ids, include):
        self.get_call = {"ids": ids, "include": include}
        return {
            "ids": ["demo.mp4::segment-b_text", "demo.mp4::segment-a_text"],
            "metadatas": [
                {"title": "B", "summary": "shared text and visual match"},
                {"title": "A", "summary": "text-only match"},
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
        },
        {
            "query_embeddings": [[0.25, 0.75]],
            "n_results": 6,
            "where": {"$and": [{"type": "visual"}, {"video_filename": "demo.mp4"}]},
        },
    ]
    assert collection.get_call == {
        "ids": ["demo.mp4::segment-b_text", "demo.mp4::segment-a_text"],
        "include": ["metadatas"],
    }
