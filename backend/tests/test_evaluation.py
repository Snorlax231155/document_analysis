import pytest
from app.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank, calculate_mrr
from app.evaluation.generation_metrics import evaluate_faithfulness, evaluate_answer_relevance


def test_recall_at_k():
    retrieved = [10, 20, 30, 40, 50]
    ground_truth = {30}

    assert recall_at_k(retrieved, ground_truth, k=1) == 0.0
    assert recall_at_k(retrieved, ground_truth, k=3) == 1.0
    assert recall_at_k(retrieved, ground_truth, k=5) == 1.0


def test_reciprocal_rank():
    retrieved = [10, 20, 30, 40]
    ground_truth = {20}
    assert reciprocal_rank(retrieved, ground_truth) == 0.5

    assert reciprocal_rank(retrieved, {99}) == 0.0


def test_mrr():
    all_ret = [[10, 20], [30, 40], [50, 60]]
    all_gt = [{10}, {40}, {99}]  # ranks: 1 (1.0), 2 (0.5), none (0.0) -> avg = 1.5 / 3 = 0.5
    assert calculate_mrr(all_ret, all_gt) == 0.5


def test_generation_metrics():
    context = [{"text": "Employees are entitled to 20 days of paid annual leave."}]
    answer = "Employees get 20 days of annual leave."

    faith = evaluate_faithfulness(answer, context)
    assert faith > 0.7

    rel = evaluate_answer_relevance("What is the annual leave policy?", answer)
    assert rel > 0.5
