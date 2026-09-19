from typing import List, Set


def recall_at_k(retrieved_ids: List[int], ground_truth_ids: Set[int], k: int) -> float:
    """
    Calculate Recall@K: Returns 1.0 if any ground truth chunk appears in top K, else 0.0.
    """
    if not ground_truth_ids:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    matched = top_k_retrieved.intersection(ground_truth_ids)
    return 1.0 if len(matched) > 0 else 0.0


def reciprocal_rank(retrieved_ids: List[int], ground_truth_ids: Set[int]) -> float:
    """
    Calculate Reciprocal Rank (RR): 1 / rank of first relevant item in retrieved list.
    """
    if not ground_truth_ids:
        return 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in ground_truth_ids:
            return 1.0 / rank

    return 0.0


def calculate_mrr(all_retrieved: List[List[int]], all_ground_truth: List[Set[int]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) across multiple query evaluation samples.
    """
    if not all_retrieved or not all_ground_truth:
        return 0.0

    rr_scores = [
        reciprocal_rank(ret, gt) for ret, gt in zip(all_retrieved, all_ground_truth)
    ]
    return sum(rr_scores) / len(rr_scores)
