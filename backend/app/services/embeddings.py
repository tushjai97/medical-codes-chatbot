"""
Embedding Generation Service
Uses Voyage AI's hosted embedding API
"""

import time
import voyageai
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

# Voyage API limits for voyage-4-lite
MAX_BATCH_SIZE = 1000
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


class EmbeddingService:
    """
    Manages embedding generation using the Voyage AI API

    Uses voyage-4-lite for fast, high quality hosted embeddings:
    - 1024 dimensions (default)
    - Network call per request (not local/free like sentence-transformers)
    - input_type distinguishes queries from documents, which matters
      for retrieval quality with Voyage's models
    """

    def __init__(self, api_key: str, model_name: str = "voyage-4-lite", dimension: int = 1024):
        """
        Initialize the Voyage embedding client

        Args:
            api_key: Voyage AI API key
            model_name: Voyage model identifier
            dimension: Expected output embedding dimension
        """
        logger.info(f"Initializing Voyage embedding client with model: {model_name}")
        self.client = voyageai.Client(api_key=api_key)
        self.model_name = model_name
        self.dimension = dimension

    def _embed_with_retry(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Call Voyage's embed API with basic retry/backoff on transient errors"""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                result = self.client.embed(
                    texts,
                    model=self.model_name,
                    input_type=input_type,
                )
                return result.embeddings
            except Exception as e:
                last_error = e
                wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                logger.warning(
                    f"Voyage embed call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
        raise last_error

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single search query

        Args:
            text: Input query text to embed

        Returns:
            Embedding vector (numpy array)
        """
        embeddings = self._embed_with_retry([text], input_type="query")
        return np.array(embeddings[0])

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = MAX_BATCH_SIZE,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple document texts (e.g. code descriptions)

        Args:
            texts: List of input texts
            batch_size: Number of texts per Voyage API request (max 1000)

        Returns:
            Array of embeddings, shape (len(texts), dimension)
        """
        batch_size = min(batch_size, MAX_BATCH_SIZE)
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            chunk_embeddings = self._embed_with_retry(chunk, input_type="document")
            all_embeddings.extend(chunk_embeddings)
            logger.info(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} texts")
        return np.array(all_embeddings)


# Global instance (singleton pattern)
# Reuse the same Voyage client for all requests
_embedding_service = None


def get_embedding_service(
    api_key: str = None,
    model_name: str = "voyage-4-lite",
    dimension: int = 1024,
) -> EmbeddingService:
    """
    Get or create embedding service singleton

    Args:
        api_key: Voyage AI API key (required on first call)
        model_name: Model to use (defaults to config)
        dimension: Expected embedding dimension

    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        if api_key is None:
            raise ValueError("api_key is required to initialize the embedding service")
        _embedding_service = EmbeddingService(api_key, model_name, dimension)
    return _embedding_service
