from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.repository import EvaluationRepository
from app.evaluation.evaluator import RAGEvaluator
from app.schemas.evaluation import EvaluationResultSchema, EvaluationRunRequestSchema

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.post("/run", response_model=EvaluationResultSchema)
def run_evaluation(
    body: Optional[EvaluationRunRequestSchema] = None,
    db: Session = Depends(get_db)
):
    """
    Run evaluation benchmark over dataset.
    Computes Recall@1, Recall@3, Recall@5, Recall@10, MRR, Faithfulness, and Answer Relevance.
    """
    try:
        evaluator = RAGEvaluator(db=db)
        dataset_path = body.sample_dataset_path if body else None
        results = evaluator.run_evaluation(dataset_path=dataset_path)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.get("/results", response_model=Optional[EvaluationResultSchema])
def get_latest_evaluation(db: Session = Depends(get_db)):
    """Retrieve the latest evaluation benchmark results."""
    repo = EvaluationRepository(db)
    latest = repo.get_latest_run()
    if not latest:
        # If no run exists in DB, execute benchmark on startup
        evaluator = RAGEvaluator(db=db)
        return evaluator.run_evaluation()
    return latest
