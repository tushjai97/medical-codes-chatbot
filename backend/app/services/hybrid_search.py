"""
Hybrid Search Service
Combines vector and keyword search using Reciprocal Rank Fusion

This is the core innovation of our system:
- Vector search: Finds semantically similar codes
- Keyword search: Catches exact medical terms
- RRF: Intelligently combines both for best results
"""

import asyncio
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from .vector_search import search_cpt_codes_vector, search_icd10_codes_vector
from .keyword_search import search_cpt_codes_keyword, search_icd10_codes_keyword
from .ranking import reciprocal_rank_fusion, normalize_scores
from .embeddings import get_embedding_service
from ..utils.pipeline_logger import log_stage, timed_stage

logger = logging.getLogger(__name__)


async def _timed(coro, stage: str, key_field: str, **extra_fields):
    """Await coro, logging its own actual duration (not the duration of
    whatever else is running concurrently alongside it via asyncio.gather)"""
    start = time.perf_counter()
    try:
        result = await coro
    except Exception as e:
        log_stage(stage, duration_ms=round((time.perf_counter() - start) * 1000, 1), error=str(e), **extra_fields)
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    log_stage(
        stage,
        duration_ms=round(duration_ms, 1),
        count=len(result),
        top=[r[key_field] for r in result[:5]],
        **extra_fields,
    )
    return result


async def hybrid_search_cpt(
    query: str,
    query_embedding: np.ndarray,
    limit: int = 10,
    category: Optional[str] = None
) -> List[Dict]:
    """
    Hybrid search for CPT codes
    Combines vector and keyword search for better accuracy

    Performance:
    - Both searches run in parallel (no time penalty)
    - Total time: max(vector_time, keyword_time) + fusion_time
    - Typically: ~100ms vector + ~10ms keyword = ~100ms total

    Args:
        query: Text query for keyword search
        query_embedding: Embedding vector for semantic search
        limit: Number of final results
        category: Optional category filter

    Returns:
        Combined ranked results with confidence scores
    """
    # Run both searches in parallel for speed
    vector_results, keyword_results = await asyncio.gather(
        _timed(search_cpt_codes_vector(query_embedding, limit=20, category=category), "vector_search", "cpt_code", code_type="cpt"),
        _timed(search_cpt_codes_keyword(query, limit=20, category=category), "keyword_search", "cpt_code", code_type="cpt"),
        return_exceptions=True  # Don't fail if one search errors
    )

    # Handle errors gracefully
    if isinstance(vector_results, Exception):
        logger.error(f"Vector search failed: {vector_results}")
        vector_results = []

    if isinstance(keyword_results, Exception):
        logger.error(f"Keyword search failed: {keyword_results}")
        keyword_results = []

    # Combine using RRF
    combined = reciprocal_rank_fusion(
        [vector_results, keyword_results],
        key_field='cpt_code'
    )

    # Normalize scores to 0-1 range
    combined = normalize_scores(combined, score_field='rrf_score')

    log_stage(
        "rrf_fusion",
        code_type="cpt",
        combined_count=len(combined),
        top=[r['cpt_code'] for r in combined[:limit]],
    )

    # Return top results
    return combined[:limit]


async def hybrid_search_icd10(
    query: str,
    query_embedding: np.ndarray,
    limit: int = 10,
    chapter: Optional[str] = None
) -> List[Dict]:
    """
    Hybrid search for ICD-10 codes
    Combines vector and keyword search for better accuracy

    Example query: "type 2 diabetes"
    - Vector search finds: E11.9, E11.65, E11.8 (semantic match)
    - Keyword search finds: E11.9, E11.8, E11.21 (exact "type 2")
    - RRF combines: E11.9 (in both!), E11.8, E11.65, E11.21

    Args:
        query: Text query for keyword search
        query_embedding: Embedding vector for semantic search
        limit: Number of final results
        chapter: Optional chapter filter

    Returns:
        Combined ranked results with confidence scores
    """
    # Run both searches in parallel
    vector_results, keyword_results = await asyncio.gather(
        _timed(search_icd10_codes_vector(query_embedding, limit=20, chapter=chapter), "vector_search", "icd10_code", code_type="icd10"),
        _timed(search_icd10_codes_keyword(query, limit=20, chapter=chapter), "keyword_search", "icd10_code", code_type="icd10"),
        return_exceptions=True
    )

    # Handle errors
    if isinstance(vector_results, Exception):
        logger.error(f"Vector search failed: {vector_results}")
        vector_results = []

    if isinstance(keyword_results, Exception):
        logger.error(f"Keyword search failed: {keyword_results}")
        keyword_results = []

    # Combine using RRF
    combined = reciprocal_rank_fusion(
        [vector_results, keyword_results],
        key_field='icd10_code'
    )

    # Normalize scores
    combined = normalize_scores(combined, score_field='rrf_score')

    log_stage(
        "rrf_fusion",
        code_type="icd10",
        combined_count=len(combined),
        top=[r['icd10_code'] for r in combined[:limit]],
    )

    # Return top results
    return combined[:limit]


async def search_all(
    query: str,
    limit_per_type: int = 5,
    filter_category: Optional[str] = None,
    filter_chapter: Optional[str] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Search both CPT and ICD-10 codes in one call

    This is the main entry point for code search:
    1. Generate embedding once
    2. Search CPT and ICD-10 in parallel
    3. Return ranked results

    Performance breakdown:
    - Embedding generation: ~15ms
    - CPT search: ~100ms
    - ICD-10 search: ~100ms
    - Total (parallel): ~115ms

    Args:
        query: Clinical description
        limit_per_type: Results per code type
        filter_category: Optional CPT category filter
        filter_chapter: Optional ICD-10 chapter filter

    Returns:
        Tuple of (cpt_results, icd10_results)
    """
    # Generate embedding once for both searches
    # Voyage's client is synchronous/network-bound, so run it off the event loop
    embedding_service = get_embedding_service()
    with timed_stage("embedding", model=embedding_service.model_name) as et:
        query_embedding = await asyncio.to_thread(embedding_service.generate_embedding, query)
        et["dimension"] = len(query_embedding)

    # Search both code types in parallel
    cpt_results, icd10_results = await asyncio.gather(
        hybrid_search_cpt(
            query,
            query_embedding,
            limit=limit_per_type,
            category=filter_category
        ),
        hybrid_search_icd10(
            query,
            query_embedding,
            limit=limit_per_type,
            chapter=filter_chapter
        )
    )

    return cpt_results, icd10_results
