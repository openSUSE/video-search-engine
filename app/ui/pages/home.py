"""Demo landing page — orient first-time viewers in 30 seconds."""

import streamlit as st

st.title("🎬 Semantic Video Search Engine")
st.caption("Multimodal ingestion + hybrid retrieval, end-to-end.")

st.markdown(
    """
This demo walks the full pipeline from raw video to searchable vector index:

1. **Submit** — pick a video that's already on disk under `data/videos/`
   and publish an ingestion job to RabbitMQ.
2. **Pipeline** — watch the worker turn the video into per-shot
   transcripts, audio events, visual captions, and action labels;
   segment them; enrich with Gemini; and write embeddings to ChromaDB.
3. **Search** — query the indexed segments with hybrid (text + visual)
   retrieval and jump the player to the matching timestamp.

> The pipeline page polls the artifact directory every 1.5 s and
> reports per-step progress with live elapsed times. No log scraping,
> no docker.sock — purely state on disk plus a Chroma probe.
"""
)

with st.container(border=True):
    st.subheader("What's running underneath")
    st.markdown(
        """
- **API** at `:1234` — FastAPI hybrid search + Prometheus `/metrics`,
  request IDs, and JSON logs.
- **Streamlit** at `:8501` — this UI.
- **Worker** — listens on RabbitMQ queue `video.ingestion`; failures
  land in the DLQ at `video.ingestion.dlq`.
- **ChromaDB** at `:8000` — vector store with cosine similarity over
  normalized 384-dim embeddings (`all-MiniLM-L6-v2`).
- **RabbitMQ management** at `:15672` — inspect the queue and DLQ.

Every search request mints an `X-Request-ID` you can grep for across
all four services.
"""
    )

st.info(
    "Start at **1 · Submit** in the sidebar.",
    icon="👈",
)
