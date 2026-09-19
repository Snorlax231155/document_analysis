import time
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging import logger, log_execution_time
from app.embeddings.embedder import EmbeddingService
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.keyword_search import KeywordSearchRetriever
from app.retrieval.hybrid_search import HybridSearchRetriever
from app.retrieval.reranker import Reranker
from app.langchain.chain import LangChainRAGChain
from app.db.repository import QueryRepository


class RAGPipeline:
    """
    Central RAG Orchestrator executing the complete pipeline:
    Native Query Embedding -> Native FAISS Search + Native SQL Keyword Search -> Native RRF Fusion -> Native Reranking -> LangChain LCEL Runnable Chain -> Native Citation Attachment.
    """

    def __init__(
        self,
        db: Session,
        vector_store: Optional[FAISSVectorStore] = None,
        embedder: Optional[EmbeddingService] = None,
        reranker: Optional[Reranker] = None,
        langchain_chain: Optional[LangChainRAGChain] = None
    ):
        self.db = db
        self.embedder = embedder or EmbeddingService()
        self.vector_store = vector_store or FAISSVectorStore()
        self.keyword_retriever = KeywordSearchRetriever(db=db)
        self.hybrid_retriever = HybridSearchRetriever()
        self.reranker = reranker or Reranker()
        self.langchain_chain = langchain_chain or LangChainRAGChain()
        self.query_repo = QueryRepository(db=db)

    def answer(
        self,
        question: str,
        top_k: int = None,
        enable_reranking: bool = None
    ) -> Dict[str, Any]:
        """
        Execute end-to-end RAG pipeline for a given user question.
        """
        start_total = time.perf_counter()
        top_k = top_k or settings.TOP_K
        final_top_k = settings.FINAL_TOP_K
        enable_rerank = enable_reranking if enable_reranking is not None else settings.ENABLE_RERANKING

        # 1. Log query to DB
        query_record = self.query_repo.create_query(
            question=question,
            top_k=top_k,
            enable_reranking=enable_rerank
        )

        start_retrieval = time.perf_counter()

        # 2. Native Semantic Search (FAISS)
        query_vec = self.embedder.embed_query(question)
        semantic_results = self.vector_store.search(query_vec, top_k=top_k)

        # 3. Native Keyword Search (SQL)
        keyword_results = self.keyword_retriever.search(question, top_k=top_k)

        # 4. Native Hybrid RRF Fusion
        hybrid_candidates = self.hybrid_retriever.fuse_results(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            top_k=top_k
        )

        # 5. Native Reranking (CrossEncoder)
        if enable_rerank:
            final_context = self.reranker.rerank(
                query=question,
                candidates=hybrid_candidates,
                top_k=final_top_k
            )
        else:
            final_context = hybrid_candidates[:final_top_k]

        retrieval_ms = round((time.perf_counter() - start_retrieval) * 1000.0, 2)

        start_gen = time.perf_counter()

        # 6. LangChain LCEL Generation & Document Orchestration
        answer_text, answer_found, langchain_docs = self.langchain_chain.run(
            question=question,
            retrieved_chunks=final_context
        )

        generation_ms = round((time.perf_counter() - start_gen) * 1000.0, 2)
        total_ms = round((time.perf_counter() - start_total) * 1000.0, 2)

        # 7. Native Citation Attachment from original metadata
        citations = []
        if answer_found:
            citations = self._extract_citations_from_langchain_docs(langchain_docs)

        # 8. Persist Response to DB
        self.query_repo.save_response(
            query_id=query_record.id,
            answer=answer_text,
            answer_found=answer_found,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            citations_data=citations
        )

        logger.info(
            f"RAG Pipeline finished query_id={query_record.id} in {total_ms} ms (Retrieval: {retrieval_ms} ms, LangChain LLM: {generation_ms} ms)"
        )

        return {
            "query_id": query_record.id,
            "question": question,
            "answer": answer_text,
            "answer_found": answer_found,
            "sources": citations,
            "retrieval_metadata": {
                "semantic_count": len(semantic_results),
                "keyword_count": len(keyword_results),
                "hybrid_candidate_count": len(hybrid_candidates),
                "final_context_count": len(final_context),
                "reranking_enabled": enable_rerank,
                "retrieved_chunks": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "document": c.get("document_name"),
                        "page": c.get("page_number"),
                        "rrf_score": c.get("rrf_score"),
                        "rerank_score": c.get("rerank_score"),
                        "text": c.get("text", "")[:120] + "..."
                    }
                    for c in final_context
                ]
            },
            "performance": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms
            }
        }

    def _extract_citations_from_langchain_docs(self, langchain_docs) -> list:
        """Deduplicate source page citations from LangChain Document metadata."""
        seen = set()
        citations = []

        for doc in langchain_docs:
            meta = doc.metadata
            doc_id = meta.get("document_id", 0)
            doc_name = meta.get("document_name", "Unknown Document")
            page_num = meta.get("page_number", 1)
            chunk_id = meta.get("chunk_id", 0)
            score = meta.get("rerank_score", meta.get("rrf_score", 0.0))

            key = (doc_id, page_num)
            if key not in seen:
                seen.add(key)
                citations.append({
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "page_number": page_num,
                    "chunk_id": chunk_id,
                    "score": round(score, 4)
                })

        return citations
