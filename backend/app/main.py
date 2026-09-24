"""
FastAPI Application - Medical Coding RAG System
Main entry point for the API server
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
from contextlib import asynccontextmanager

from .config import settings
from .database import db
from .models.request_models import CodingQuery
from .models.response_models import (
    CodingResponse, CodeSuggestion, StatsResponse, HealthResponse
)
from .services.hybrid_search import search_all
from .services.reranker_service import get_reranker_service
from .services.embeddings import get_embedding_service
from .utils.logger import setup_logging

# Setup logging
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events

    Startup:
    - Connect to database
    - Load embedding model

    Shutdown:
    - Close database connections
    """
    # Startup
    logger.info("Starting Medical Coding RAG API...")

    # Connect to database
    await db.connect()
    logger.info("Database connected")

    # Initialize embedding client (pre-initializes to avoid first-request delay)
    embedding_service = get_embedding_service(
        api_key=settings.VOYAGE_API_KEY,
        model_name=settings.VOYAGE_MODEL_NAME,
        dimension=settings.EMBEDDING_DIM,
    )
    logger.info(f"Embedding model loaded: {settings.VOYAGE_MODEL_NAME}")

    logger.info("API ready!")

    yield

    # Shutdown
    logger.info("Shutting down API...")
    await db.disconnect()
    logger.info("API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Medical Coding RAG API",
    description="""
    Retrieval Augmented Generation system for medical code suggestions.

    ## Features
    - **Hybrid Search**: Combines vector similarity + keyword matching
    - **Three Modes**: Quick (no LLM), Standard (cached), Expert (full LLM)
    - **Real 2025 Data**: 1,164 CPT codes + 74,260 ICD-10 codes

    ## Search Modes
    - **Quick**: Hybrid search only (~200ms)
    - **Standard**: Cached results (~100ms) or hybrid search
    - **Expert**: LLM reranking with explanations (~2s)
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Medical Coding RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "stats": "/api/stats"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Tests:
    - Database connection
    - Returns embedding model info
    """
    try:
        # Test database connection
        count = await db.fetchval("SELECT COUNT(*) FROM cpt_codes")
        db_status = "connected"
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        db_status = f"error: {str(e)}"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "unhealthy",
        database=db_status,
        embedding_model=settings.VOYAGE_MODEL_NAME
    )


@app.get("/api/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats():
    """
    Get database statistics

    Returns:
    - Total CPT codes
    - Total ICD-10 codes
    - Available categories and chapters
    """
    try:
        # Count totals
        cpt_count = await db.fetchval("SELECT COUNT(*) FROM cpt_codes")
        icd10_count = await db.fetchval("SELECT COUNT(*) FROM icd10_codes")

        # Get unique categories
        categories = await db.fetch(
            "SELECT DISTINCT category FROM cpt_codes WHERE category IS NOT NULL ORDER BY category"
        )

        # Get unique chapters
        chapters = await db.fetch(
            "SELECT DISTINCT chapter FROM icd10_codes WHERE chapter IS NOT NULL ORDER BY chapter"
        )

        return StatsResponse(
            total_cpt_codes=cpt_count,
            total_icd10_codes=icd10_count,
            categories=[row['category'] for row in categories],
            chapters=[row['chapter'] for row in chapters]
        )

    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/code-suggestions",
    response_model=CodingResponse,
    tags=["Search"]
)
async def get_code_suggestions(query: CodingQuery):
    """
    Get medical code suggestions for a clinical description

    ## Search Process

    ### Quick Mode (fastest, ~200ms)
    1. Generate embedding
    2. Hybrid search (vector + keyword)
    3. Reciprocal Rank Fusion
    4. Return top results

    ### Standard Mode (default, ~100-500ms)
    - Same as Quick mode
    - Future: Check cache first

    ### Expert Mode (most detailed, ~2s)
    - All Quick mode steps
    - Send to LLM for reranking
    - Add detailed reasoning
    - Include overall explanation

    ## Example Request
    ```json
    {
      "clinical_description": "patient with type 2 diabetes",
      "max_results": 5,
      "search_mode": "quick"
    }
    ```

    ## Example Response
    ```json
    {
      "query": "patient with type 2 diabetes",
      "cpt_codes": [...],
      "icd10_codes": [...],
      "processing_time_ms": 234.5
    }
    ```
    """
    start_time = time.time()

    try:
        # Perform hybrid search
        cpt_results, icd10_results = await search_all(
            query.clinical_description,
            limit_per_type=query.max_results,
            filter_category=query.filter_category,
            filter_chapter=query.filter_chapter
        )

        # Mode-based processing
        if query.search_mode == "expert":
            # Local cross-encoder reranking (no LLM/API call)
            logger.info(f"Using Expert mode with reranker for query: {query.clinical_description[:50]}...")

            reranker_service = get_reranker_service(settings.RERANKER_MODEL_NAME)
            rerank_results = await reranker_service.rerank_codes(
                query.clinical_description,
                cpt_results,
                icd10_results
            )

            # Map reranked results back to full code details
            cpt_map = {c['cpt_code']: c for c in cpt_results}
            icd10_map = {c['icd10_code']: c for c in icd10_results}

            cpt_codes = [
                CodeSuggestion(
                    code=item['code'],
                    description=cpt_map[item['code']]['description'],
                    code_type="CPT",
                    category=cpt_map[item['code']].get('category'),
                    confidence_score=item['confidence'],
                    reasoning=item.get('reasoning')
                )
                for item in rerank_results['cpt_codes']
                if item['code'] in cpt_map
            ]

            icd10_codes = [
                CodeSuggestion(
                    code=item['code'],
                    description=icd10_map[item['code']]['description'],
                    code_type="ICD-10",
                    category=icd10_map[item['code']].get('chapter'),
                    confidence_score=item['confidence'],
                    reasoning=item.get('reasoning')
                )
                for item in rerank_results['icd10_codes']
                if item['code'] in icd10_map
            ]

            explanation = rerank_results.get('explanation')

        else:
            # Quick/Standard mode (no LLM)
            logger.info(f"Using {query.search_mode} mode for query: {query.clinical_description[:50]}...")

            cpt_codes = [
                CodeSuggestion(
                    code=item['cpt_code'],
                    description=item['description'],
                    code_type="CPT",
                    category=item.get('category'),
                    confidence_score=item['confidence_score'],
                    reasoning=None
                )
                for item in cpt_results
            ]

            icd10_codes = [
                CodeSuggestion(
                    code=item['icd10_code'],
                    description=item['description'],
                    code_type="ICD-10",
                    category=item.get('chapter'),
                    confidence_score=item['confidence_score'],
                    reasoning=None
                )
                for item in icd10_results
            ]

            explanation = None

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Query completed in {processing_time:.1f}ms: "
            f"{len(cpt_codes)} CPT, {len(icd10_codes)} ICD-10"
        )

        return CodingResponse(
            query=query.clinical_description,
            cpt_codes=cpt_codes,
            icd10_codes=icd10_codes,
            search_mode=query.search_mode,
            processing_time_ms=processing_time,
            explanation=explanation
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
