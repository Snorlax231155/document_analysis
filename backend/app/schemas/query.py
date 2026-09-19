from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class QueryRequestSchema(BaseModel):
    question: str = Field(..., min_length=2, description="Natural language question to ask document assistant")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Top K candidate chunks to retrieve")
    enable_reranking: Optional[bool] = Field(True, description="Enable CrossEncoder reranking stage")


class CitationSchema(BaseModel):
    document_id: int
    document_name: str
    page_number: int
    chunk_id: Optional[int] = 0
    score: Optional[float] = 0.0


class RetrievalMetadataSchema(BaseModel):
    semantic_count: int
    keyword_count: int
    hybrid_candidate_count: int
    final_context_count: int
    reranking_enabled: bool
    retrieved_chunks: List[Dict[str, Any]] = []


class PerformanceMetricsSchema(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class QueryResponseSchema(BaseModel):
    query_id: int
    question: str
    answer: str
    answer_found: bool
    sources: List[CitationSchema] = []
    retrieval_metadata: RetrievalMetadataSchema
    performance: PerformanceMetricsSchema
