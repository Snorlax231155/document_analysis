from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.repository import QueryRepository
from app.rag.pipeline import RAGPipeline
from app.schemas.query import QueryRequestSchema, QueryResponseSchema

router = APIRouter(prefix="", tags=["Query"])


@router.post("/query", response_model=QueryResponseSchema)
def submit_query(
    body: QueryRequestSchema,
    db: Session = Depends(get_db)
):
    """
    Execute hybrid RAG pipeline query over uploaded documents.
    Returns grounded LLM answer, source citations, retrieval breakdown, and performance timing.
    """
    try:
        pipeline = RAGPipeline(db=db)
        result = pipeline.answer(
            question=body.question,
            top_k=body.top_k,
            enable_reranking=body.enable_reranking
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


@router.get("/queries", response_model=List[Dict[str, Any]])
def get_query_history(limit: int = 20, db: Session = Depends(get_db)):
    """Retrieve recent query history and response statistics."""
    repo = QueryRepository(db)
    return repo.get_recent_queries(limit=limit)
