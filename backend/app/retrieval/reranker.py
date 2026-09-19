from typing import List, Dict, Any
from app.core.config import settings
from app.core.logging import logger, log_execution_time


class Reranker:
    """
    Cross-Encoder Reranker for refine-stage context ranking.
    Uses sentence-transformers CrossEncoder if available, with robust lexical fallback.
    """

    def __init__(self, model_name: str = None, enabled: bool = None):
        self.enabled = enabled if enabled is not None else settings.ENABLE_RERANKING
        self.model_name = model_name or settings.RERANK_MODEL
        self._cross_encoder = None

    def _get_cross_encoder(self):
        """Lazy load CrossEncoder model."""
        if self._cross_encoder is None and self.enabled:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder model: {self.model_name}")
                self._cross_encoder = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder ({e}). Falling back to lexical reranker.")
                self._cross_encoder = False
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank hybrid search candidate chunks for a given query.
        """
        if not candidates or not self.enabled:
            return candidates[:top_k]

        with log_execution_time(f"Reranking {len(candidates)} Candidates (top_k={top_k})"):
            encoder = self._get_cross_encoder()

            if encoder:
                try:
                    pairs = [[query, candidate["text"]] for candidate in candidates]
                    scores = encoder.predict(pairs)
                    for candidate, score in zip(candidates, scores):
                        candidate["rerank_score"] = float(score)
                        candidate["score"] = float(score)
                except Exception as e:
                    logger.error(f"CrossEncoder prediction error: {e}. Using hybrid scores.")
                    self._fallback_lexical_rerank(query, candidates)
            else:
                self._fallback_lexical_rerank(query, candidates)

            # Sort by rerank score descending
            reranked = sorted(candidates, key=lambda x: x.get("rerank_score", x.get("score", 0.0)), reverse=True)
            top_reranked = reranked[:top_k]

            for rank, item in enumerate(top_reranked, start=1):
                item["final_rank"] = rank

        logger.info(f"Reranked {len(candidates)} candidates down to {len(top_reranked)} top chunks")
        return top_reranked

    def _fallback_lexical_rerank(self, query: str, candidates: List[Dict[str, Any]]) -> None:
        """Heuristic lexical overlap reranking fallback."""
        query_terms = set(query.lower().split())
        for item in candidates:
            text_terms = set(item["text"].lower().split())
            overlap = len(query_terms.intersection(text_terms))
            item["rerank_score"] = item.get("score", 0.0) + (overlap * 0.1)
