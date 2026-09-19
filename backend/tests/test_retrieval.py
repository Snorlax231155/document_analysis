import numpy as np
import pytest
from app.embeddings.embedder import EmbeddingService
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.hybrid_search import HybridSearchRetriever
from app.retrieval.reranker import Reranker


def test_embedding_service_shape():
    embedder = EmbeddingService()
    texts = ["Annual leave policy for employees", "Remote work security requirements"]
    vecs = embedder.embed_documents(texts)

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (2, 384)
    # Check L2 normalization (norm should be ~1.0)
    norm = np.linalg.norm(vecs[0])
    assert abs(norm - 1.0) < 1e-3


def test_faiss_vector_store_add_and_search(tmp_path):
    idx_path = str(tmp_path / "test_faiss.index")
    meta_path = str(tmp_path / "test_meta.json")

    store = FAISSVectorStore(index_path=idx_path, metadata_path=meta_path, embedding_dimension=384)
    embedder = EmbeddingService()

    chunks = [
        {"chunk_id": 1, "document_id": 1, "document_name": "hr.pdf", "page_number": 1, "chunk_index": 0, "text": "Employees get 20 days annual leave."},
        {"chunk_id": 2, "document_id": 1, "document_name": "hr.pdf", "page_number": 2, "chunk_index": 1, "text": "Remote work security policy code SEC-402."}
    ]

    embeddings = embedder.embed_documents([c["text"] for c in chunks])
    store.add_documents(chunks, embeddings)

    assert store.index.ntotal == 2

    # Query search
    q_vec = embedder.embed_query("annual leave days")
    results = store.search(q_vec, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == 1
    assert "score" in results[0]


def test_rrf_hybrid_fusion():
    fusion = HybridSearchRetriever(rrf_k=60)
    semantic_res = [
        {"chunk_id": 10, "text": "Leave policy", "semantic_rank": 1, "semantic_score": 0.9},
        {"chunk_id": 20, "text": "Remote work", "semantic_rank": 2, "semantic_score": 0.7}
    ]
    keyword_res = [
        {"chunk_id": 20, "text": "Remote work", "keyword_rank": 1, "keyword_score": 5.0},
        {"chunk_id": 30, "text": "Travel expense", "keyword_rank": 2, "keyword_score": 3.0}
    ]

    fused = fusion.fuse_results(semantic_res, keyword_res, top_k=3)
    assert len(fused) == 3
    # Chunk 20 matched in both semantic & keyword, so it should rank first with highest RRF score
    assert fused[0]["chunk_id"] == 20
    assert "rrf_score" in fused[0]


def test_reranker_fallback():
    reranker = Reranker(enabled=True)
    candidates = [
        {"chunk_id": 1, "text": "Generic company overview with no specifics.", "score": 0.5},
        {"chunk_id": 2, "text": "Specific annual leave policy details for employees.", "score": 0.4}
    ]

    reranked = reranker.rerank("annual leave policy", candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == 2
