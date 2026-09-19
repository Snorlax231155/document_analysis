from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.models import Document, Chunk, Query, Response, Citation, EvaluationRun


class DocumentRepository:
    """Repository handling database operations for documents and chunks."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_hash(self, file_hash: str) -> Optional[Document]:
        return self.db.query(Document).filter(Document.file_hash == file_hash).first()

    def get_by_id(self, document_id: int) -> Optional[Document]:
        return self.db.query(Document).filter(Document.id == document_id).first()

    def list_all(self) -> List[Document]:
        return self.db.query(Document).order_by(Document.upload_time.desc()).all()

    def create_document(
        self,
        filename: str,
        file_path: str,
        file_hash: str,
        file_size: int,
        page_count: int
    ) -> Document:
        doc = Document(
            filename=filename,
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
            page_count=page_count,
            chunk_count=0,
            status="processing"
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def save_chunks(self, document_id: int, chunks_data: List[Dict[str, Any]]) -> List[Chunk]:
        chunk_objects = []
        for c in chunks_data:
            chunk_obj = Chunk(
                document_id=document_id,
                page_number=c["page_number"],
                chunk_index=c["chunk_index"],
                text=c["text"],
                token_count=c.get("token_count", len(c["text"]) // 4)
            )
            self.db.add(chunk_obj)
            chunk_objects.append(chunk_obj)

        doc = self.get_by_id(document_id)
        if doc:
            doc.chunk_count = len(chunks_data)
            doc.status = "processed"

        self.db.commit()
        for chunk_obj in chunk_objects:
            self.db.refresh(chunk_obj)
        return chunk_objects

    def delete_document(self, document_id: int) -> bool:
        doc = self.get_by_id(document_id)
        if not doc:
            return False
        self.db.delete(doc)
        self.db.commit()
        return True

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Fetch all chunks across all documents with joined metadata."""
        results = (
            self.db.query(Chunk, Document.filename)
            .join(Document, Chunk.document_id == Document.id)
            .all()
        )
        chunks = []
        for chunk, filename in results:
            chunks.append({
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "document_name": filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "token_count": chunk.token_count
            })
        return chunks


class QueryRepository:
    """Repository handling queries, responses, and citations."""

    def __init__(self, db: Session):
        self.db = db

    def create_query(self, question: str, top_k: int, enable_reranking: bool) -> Query:
        query_obj = Query(
            question=question,
            top_k=top_k,
            enable_reranking=enable_reranking
        )
        self.db.add(query_obj)
        self.db.commit()
        self.db.refresh(query_obj)
        return query_obj

    def save_response(
        self,
        query_id: int,
        answer: str,
        answer_found: bool,
        retrieval_ms: float,
        generation_ms: float,
        total_ms: float,
        citations_data: List[Dict[str, Any]]
    ) -> Response:
        resp = Response(
            query_id=query_id,
            answer=answer,
            answer_found=answer_found,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms
        )
        self.db.add(resp)
        self.db.commit()
        self.db.refresh(resp)

        for cite in citations_data:
            c_obj = Citation(
                response_id=resp.id,
                document_id=cite["document_id"],
                page_number=cite["page_number"],
                chunk_id=cite.get("chunk_id", 0),
                document_name=cite["document_name"],
                relevance_score=cite.get("score", cite.get("relevance_score", 0.0))
            )
            self.db.add(c_obj)

        self.db.commit()
        self.db.refresh(resp)
        return resp

    def get_recent_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        results = (
            self.db.query(Query)
            .order_by(Query.timestamp.desc())
            .limit(limit)
            .all()
        )
        data = []
        for q in results:
            resp = q.responses[0] if q.responses else None
            data.append({
                "query_id": q.id,
                "question": q.question,
                "timestamp": q.timestamp.isoformat(),
                "answer": resp.answer if resp else None,
                "answer_found": resp.answer_found if resp else False,
                "total_ms": resp.total_ms if resp else 0.0,
                "citations_count": len(resp.citations) if resp else 0
            })
        return data


class EvaluationRepository:
    """Repository handling evaluation run logs."""

    def __init__(self, db: Session):
        self.db = db

    def save_evaluation_run(
        self,
        recall_at_1: float,
        recall_at_3: float,
        recall_at_5: float,
        recall_at_10: float,
        mrr: float,
        faithfulness: float,
        answer_relevance: float,
        total_samples: int
    ) -> EvaluationRun:
        eval_run = EvaluationRun(
            recall_at_1=recall_at_1,
            recall_at_3=recall_at_3,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            mrr=mrr,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            total_samples=total_samples
        )
        self.db.add(eval_run)
        self.db.commit()
        self.db.refresh(eval_run)
        return eval_run

    def get_latest_run(self) -> Optional[EvaluationRun]:
        return self.db.query(EvaluationRun).order_by(EvaluationRun.timestamp.desc()).first()
