# ingestion_pipeline/steps/step_04_indexing.py

import logging
import json
import os
from sentence_transformers import SentenceTransformer
import chromadb
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def _prepare_metadata_for_db(segment: Dict[str, Any], video_filename: str) -> Dict[str, Any]:
    """
    Prepares a clean metadata dictionary for ChromaDB.
    ChromaDB metadata values must be strings, ints, floats, or bools.
    This metadata will be attached to BOTH the text and visual entries.
    """
    # Join lists into comma-separated strings for filtering
    speakers_str = ",".join(segment.get('speakers', []))
    keywords_str = ",".join(segment.get('keywords', []))
    actions_str = ",".join(segment.get('consolidated_actions', []))
    
    return {
        # Core searchable metadata
        "title": str(segment.get('title', '')),
        "summary": str(segment.get('summary', '')),
        "speakers": speakers_str,
        "keywords": keywords_str,
        "actions": actions_str,
        
        # Essential data for displaying results
        "start_time": float(segment.get('start_time', 0.0)),
        "end_time": float(segment.get('end_time', 0.0)),
        "video_filename": video_filename,
    }

def run_indexing(enriched_segments_path: str, video_filename: str, config: Dict[str, Any]):
    """
    Takes final enriched segments, creates separate text and visual vector embeddings,
    and indexes them in a single ChromaDB collection with distinct IDs.
    """
    logger.info("--- Starting Step 4: Indexing Segments in Vector DB ---")
    
    # 1. Load the enriched segments data
    logger.info(f"1/4: Loading enriched segments from {enriched_segments_path}...")
    with open(enriched_segments_path, 'r') as f:
        segments = json.load(f)
    
    if not segments:
        logger.warning("No segments found in the input file. Skipping indexing.")
        return

    # 2. Initialize the embedding model and ChromaDB client
    logger.info("2/4: Initializing models and DB client...")
    device = config['general']['device']
    embedding_model_name = config['models']['embedding']['name']
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    
    db_config = config['database']
    client = chromadb.HttpClient(host=db_config['host'], port=db_config['port'])
    collection = client.get_or_create_collection(
        name=db_config['collection_name'],
        metadata={"hnsw:space": "cosine"} # Using cosine similarity is standard for text
    )

    # 3. Prepare data for batch insertion
    logger.info(f"3/4: Preparing {len(segments)} segments for batch indexing...")
    
    # We will create two sets of entries: one for text, one for visuals.
    ids_to_add = []
    embeddings_to_add = []
    metadatas_to_add = []

    # Create all embeddings at once for GPU efficiency
    all_transcripts = [seg.get('full_transcript', '') or seg.get('summary', '') for seg in segments] # Fallback to summary if no transcript

    all_visual_contexts = []
    for seg in segments:
        visual_parts = [seg.get('summary', '')]
        visual_parts.extend(seg.get('consolidated_visual_captions', []))
        visual_parts.extend(seg.get('consolidated_actions', []))
        # Join non-empty parts with a period for semantic separation
        all_visual_contexts.append(". ".join(filter(None, visual_parts)))

    text_embeddings = embedding_model.encode(all_transcripts, show_progress_bar=True)
    visual_embeddings = embedding_model.encode(all_visual_contexts, show_progress_bar=True)

    for i, segment in enumerate(segments):
        segment_id = segment['segment_id']
        common_metadata = _prepare_metadata_for_db(segment, video_filename)
        
        # --- Create the TEXT entry ---
        # We create a unique ID for the text embedding of this segment.
        ids_to_add.append(f"{segment_id}_text")
        embeddings_to_add.append(text_embeddings[i].tolist())
        # Add a type identifier to the metadata for filtering
        text_metadata = common_metadata.copy()
        text_metadata['type'] = 'text'
        metadatas_to_add.append(text_metadata)
        
        # --- Create the VISUAL entry ---
        # We create a unique ID for the visual embedding of this segment.
        ids_to_add.append(f"{segment_id}_visual")
        embeddings_to_add.append(visual_embeddings[i].tolist())
        # Add a type identifier to the metadata for filtering
        visual_metadata = common_metadata.copy()
        visual_metadata['type'] = 'visual'
        metadatas_to_add.append(visual_metadata)

    # 4. Add the data to ChromaDB in a single batch operation
    # This adds 2 * num_segments entries to the database.
    logger.info(f"4/4: Indexing {len(ids_to_add)} documents into ChromaDB collection '{db_config['collection_name']}'...")
    
    # Use 'upsert' to safely add or update entries without duplication errors.
    collection.upsert(
        ids=ids_to_add,
        embeddings=embeddings_to_add,
        metadatas=metadatas_to_add
    )

    logger.info("--- Indexing Step Complete! ---")
    logger.info(f"Successfully indexed text and visual data for {len(segments)} segments.")
