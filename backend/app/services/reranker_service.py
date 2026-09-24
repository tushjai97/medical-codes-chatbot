"""
Reranker Service for Expert Mode
Uses a local open-source cross-encoder (BAAI/bge-reranker-v2-m3) instead
of an LLM API to rerank hybrid search results.

Only used in Expert mode to:
- Rerank top codes by cross-encoder relevance (query, code description)
- No network calls, no API cost, no reasoning/explanation text
  (a cross-encoder outputs a relevance score, not natural language)
"""

import asyncio
import math
from typing import List, Dict
import logging

from sentence_transformers import CrossEncoder

from ..utils.pipeline_logger import timed_stage

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
CANDIDATES_PER_TYPE = 10  # how many hybrid-search candidates to rerank
TOP_K = 5  # how many reranked results to keep per type


class RerankerService:
    """
    Local cross-encoder reranker for medical code candidates

    Uses BAAI/bge-reranker-v2-m3:
    - MIT licensed, ~568M params, runs on CPU
    - Scores (query, code description) pairs directly (cross-encoder),
      which is more accurate for this kind of short-text relevance
      ranking than embedding cosine similarity alone
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL_NAME):
        logger.info(f"Loading reranker model: {model_name}")
        self.model_name = model_name
        # Force CPU: the MPS (Apple GPU) backend isn't safe for concurrent
        # predict() calls from multiple threads and can crash the process
        self.model = CrossEncoder(model_name, device="cpu")
        logger.info("Reranker model loaded")

    def _score(self, query: str, descriptions: List[str]) -> List[float]:
        """Score (query, description) pairs, normalized to 0-1 via sigmoid"""
        if not descriptions:
            return []
        pairs = [[query, desc] for desc in descriptions]
        raw_scores = self.model.predict(pairs)
        return [1 / (1 + math.exp(-float(s))) for s in raw_scores]

    async def rerank_codes(
        self,
        query: str,
        cpt_codes: List[Dict],
        icd10_codes: List[Dict]
    ) -> Dict:
        """
        Rerank hybrid search results using the cross-encoder

        Args:
            query: Clinical description
            cpt_codes: Retrieved CPT codes (from hybrid search)
            icd10_codes: Retrieved ICD-10 codes (from hybrid search)

        Returns:
            Dict with reranked codes (same shape the API layer expects
            from the old LLM-based reranker), and no natural-language
            explanation since a cross-encoder doesn't generate one
        """
        cpt_candidates = cpt_codes[:CANDIDATES_PER_TYPE]
        icd10_candidates = icd10_codes[:CANDIDATES_PER_TYPE]

        # CrossEncoder.predict is a blocking CPU call; keep it off the event
        # loop. Score sequentially in one thread -- concurrent predict()
        # calls on the same model instance offer no speedup on CPU and
        # risk thread-safety issues.
        def _score_both():
            return (
                self._score(query, [c['description'] for c in cpt_candidates]),
                self._score(query, [c['description'] for c in icd10_candidates]),
            )

        with timed_stage("rerank", model=self.model_name) as rt:
            cpt_scores, icd10_scores = await asyncio.to_thread(_score_both)

            cpt_ranked = sorted(
                zip(cpt_candidates, cpt_scores), key=lambda x: x[1], reverse=True
            )[:TOP_K]
            icd10_ranked = sorted(
                zip(icd10_candidates, icd10_scores), key=lambda x: x[1], reverse=True
            )[:TOP_K]

            rt["cpt_reranked"] = [(c['cpt_code'], round(s, 3)) for c, s in cpt_ranked]
            rt["icd10_reranked"] = [(c['icd10_code'], round(s, 3)) for c, s in icd10_ranked]

        return {
            "cpt_codes": [
                {"code": code['cpt_code'], "confidence": score, "reasoning": None}
                for code, score in cpt_ranked
            ],
            "icd10_codes": [
                {"code": code['icd10_code'], "confidence": score, "reasoning": None}
                for code, score in icd10_ranked
            ],
            "explanation": (
                f"Reranked locally using {self.model_name} "
                "(open-source cross-encoder, no LLM used)."
            )
        }


# Global instance (singleton pattern)
_reranker_service = None


def get_reranker_service(model_name: str = DEFAULT_RERANKER_MODEL_NAME) -> RerankerService:
    """Get or create reranker service singleton (loads the model once)"""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService(model_name)
    return _reranker_service
