from typing import List, Dict, Any
from app.core.config import settings
from app.core.logging import logger, log_execution_time


class HybridSearchRetriever:
    """
    Hybrid retriever combining Semantic Vector Search (FAISS) and Keyword Search (SQL)
    using Reciprocal Rank Fusion (RRF) and linear score weighting.
    """

    def __init__(
        self,
        vector_weight: float = None,
        keyword_weight: float = None,
        rrf_k: int = 60
    ):
        self.vector_weight = vector_weight if vector_weight is not None else settings.VECTOR_WEIGHT
        self.keyword_weight = keyword_weight if keyword_weight is not None else settings.KEYWORD_WEIGHT
        self.rrf_k = rrf_k

    def fuse_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Combine semantic vector results and keyword results using Reciprocal Rank Fusion (RRF).
        
        RRF Formula:
            RRF_Score(doc) = 1 / (k + rank_semantic) + 1 / (k + rank_keyword)
        """
        with log_execution_time("Hybrid Search RRF Fusion"):
            fusion_map: Dict[int, Dict[str, Any]] = {}

            # Process Semantic Results
            for item in semantic_results:
                chunk_id = item["chunk_id"]
                rank = item.get("semantic_rank", 999)
                rrf_score = 1.0 / (self.rrf_k + rank)

                if chunk_id not in fusion_map:
                    fusion_map[chunk_id] = dict(item)
                    fusion_map[chunk_id]["rrf_score"] = 0.0
                    fusion_map[chunk_id]["sources_matched"] = []

                fusion_map[chunk_id]["rrf_score"] += self.vector_weight * rrf_score
                fusion_map[chunk_id]["semantic_rank"] = rank
                fusion_map[chunk_id]["semantic_score"] = item.get("semantic_score", 0.0)
                fusion_map[chunk_id]["sources_matched"].append("semantic")

            # Process Keyword Results
            for item in keyword_results:
                chunk_id = item["chunk_id"]
                rank = item.get("keyword_rank", 999)
                rrf_score = 1.0 / (self.rrf_k + rank)

                if chunk_id not in fusion_map:
                    fusion_map[chunk_id] = dict(item)
                    fusion_map[chunk_id]["rrf_score"] = 0.0
                    fusion_map[chunk_id]["sources_matched"] = []

                fusion_map[chunk_id]["rrf_score"] += self.keyword_weight * rrf_score
                fusion_map[chunk_id]["keyword_rank"] = rank
                fusion_map[chunk_id]["keyword_score"] = item.get("keyword_score", 0.0)
                fusion_map[chunk_id]["sources_matched"].append("keyword")

            # Convert to list and sort by RRF score descending
            fused_list = list(fusion_map.values())
            fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)

            top_fused = fused_list[:top_k]

            # Assign final hybrid rank and score
            for rank, item in enumerate(top_fused, start=1):
                item["hybrid_rank"] = rank
                item["score"] = round(item["rrf_score"], 6)

        logger.info(
            f"Hybrid RRF fusion combined {len(semantic_results)} semantic and {len(keyword_results)} keyword results into {len(top_fused)} candidates."
        )
        return top_fused
