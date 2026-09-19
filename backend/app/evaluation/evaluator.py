import json
from pathlib import Path
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging import logger, log_execution_time
from app.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank, calculate_mrr
from app.evaluation.generation_metrics import evaluate_faithfulness, evaluate_answer_relevance
from app.rag.pipeline import RAGPipeline
from app.db.repository import EvaluationRepository


class RAGEvaluator:
    """
    RAG Evaluation Engine computing empirical benchmark metrics:
    Recall@1, Recall@3, Recall@5, Recall@10, MRR, Faithfulness, Answer Relevance.
    """

    def __init__(self, db: Session, rag_pipeline: RAGPipeline = None):
        self.db = db
        self.rag_pipeline = rag_pipeline or RAGPipeline(db=db)
        self.eval_repo = EvaluationRepository(db=db)

    def run_evaluation(self, dataset_path: str = None) -> Dict[str, Any]:
        """
        Execute benchmark evaluation over JSON evaluation dataset.
        """
        path = Path(dataset_path or (settings.EVALUATION_DIR / "sample_dataset.json"))
        if not path.exists():
            logger.warning(f"Evaluation dataset not found at {path}. Returning empty metrics.")
            return self._empty_result()

        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        if not samples:
            return self._empty_result()

        r1_list, r3_list, r5_list, r10_list = [], [], [], []
        rr_list = []
        faith_list = []
        relevance_list = []

        with log_execution_time(f"RAG Evaluation Benchmark ({len(samples)} samples)"):
            for sample in samples:
                question = sample["question"]
                exp_doc = sample.get("expected_document")
                exp_text = sample.get("expected_text_contains")

                # Run RAG Pipeline
                rag_resp = self.rag_pipeline.answer(question=question, top_k=10)

                answer = rag_resp["answer"]
                retrieved_chunks = rag_resp["retrieval_metadata"]["retrieved_chunks"]

                # Identify relevant chunk indices in retrieved results
                relevant_ranks = []
                for idx, chunk in enumerate(retrieved_chunks, start=1):
                    doc_match = (not exp_doc) or (exp_doc.lower() in chunk["document"].lower())
                    text_match = (not exp_text) or (exp_text.lower() in chunk["text"].lower())
                    if doc_match and text_match:
                        relevant_ranks.append(idx)

                # Retrieval Metrics
                is_hit_1 = 1.0 if any(r <= 1 for r in relevant_ranks) else 0.0
                is_hit_3 = 1.0 if any(r <= 3 for r in relevant_ranks) else 0.0
                is_hit_5 = 1.0 if any(r <= 5 for r in relevant_ranks) else 0.0
                is_hit_10 = 1.0 if any(r <= 10 for r in relevant_ranks) else 0.0

                rr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0

                r1_list.append(is_hit_1)
                r3_list.append(is_hit_3)
                r5_list.append(is_hit_5)
                r10_list.append(is_hit_10)
                rr_list.append(rr)

                # Generation Metrics
                faith = evaluate_faithfulness(answer, retrieved_chunks)
                rel = evaluate_answer_relevance(question, answer)

                faith_list.append(faith)
                relevance_list.append(rel)

        total_samples = len(samples)
        m_r1 = round(sum(r1_list) / total_samples, 4)
        m_r3 = round(sum(r3_list) / total_samples, 4)
        m_r5 = round(sum(r5_list) / total_samples, 4)
        m_r10 = round(sum(r10_list) / total_samples, 4)
        m_mrr = round(sum(rr_list) / total_samples, 4)
        m_faith = round(sum(faith_list) / total_samples, 4)
        m_rel = round(sum(relevance_list) / total_samples, 4)

        # Save run log to DB
        eval_run = self.eval_repo.save_evaluation_run(
            recall_at_1=m_r1,
            recall_at_3=m_r3,
            recall_at_5=m_r5,
            recall_at_10=m_r10,
            mrr=m_mrr,
            faithfulness=m_faith,
            answer_relevance=m_rel,
            total_samples=total_samples
        )

        return {
            "id": eval_run.id,
            "timestamp": eval_run.timestamp.isoformat(),
            "recall_at_1": m_r1,
            "recall_at_3": m_r3,
            "recall_at_5": m_r5,
            "recall_at_10": m_r10,
            "mrr": m_mrr,
            "faithfulness": m_faith,
            "answer_relevance": m_rel,
            "total_samples": total_samples
        }

    def _empty_result() -> Dict[str, Any]:
        return {
            "timestamp": "",
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "total_samples": 0
        }
