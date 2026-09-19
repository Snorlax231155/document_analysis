import json
import os
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.logging import logger, log_execution_time


class FAISSVectorStore:
    """
    Persistent FAISS vector index with separate JSON metadata mapping.
    Uses Inner Product (IndexFlatIP) on L2 normalized vectors for Cosine Similarity.
    """

    def __init__(
        self,
        index_path: str = None,
        metadata_path: str = None,
        embedding_dimension: int = 384
    ):
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.metadata_path = metadata_path or settings.FAISS_METADATA_PATH
        self.embedding_dimension = embedding_dimension
        
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

        self._initialize_store()

    def _initialize_store(self) -> None:
        """Load index and metadata from disk if present, else create new index."""
        if Path(self.index_path).exists() and Path(self.metadata_path).exists():
            self.load()
        else:
            self._create_empty_index()

    def _create_empty_index(self) -> None:
        """Create new empty FAISS IndexFlatIP."""
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.metadata = []
        logger.info(f"Initialized new empty FAISS index (dimension={self.embedding_dimension})")

    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> None:
        """Add document chunk embeddings and metadata to the FAISS index."""
        if len(chunks) == 0 or embeddings.size == 0:
            return

        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.embedding_dimension}, got {embeddings.shape[1]}"
            )

        with log_execution_time(f"FAISS Add {len(chunks)} Chunks"):
            self.index.add(embeddings)
            self.metadata.extend(chunks)
            self.save()

        logger.info(f"Added {len(chunks)} chunks to FAISS index. Total index size: {self.index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search top_k nearest vector neighbors.
        
        Returns list of dicts: [
            {
                "chunk_id": int,
                "document_id": int,
                "document_name": str,
                "page_number": int,
                "chunk_index": int,
                "text": str,
                "score": float (cosine similarity score),
                "rank": int (1-based)
            }
        ]
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        top_k = min(top_k, self.index.ntotal)
        
        with log_execution_time(f"FAISS Search (top_k={top_k})"):
            scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = dict(self.metadata[idx])
            meta["score"] = float(score)
            meta["semantic_score"] = float(score)
            meta["semantic_rank"] = rank
            results.append(meta)

        return results

    def save(self) -> None:
        """Persist FAISS index binary and metadata JSON to disk."""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved FAISS index ({self.index.ntotal} vectors) to {self.index_path}")

    def load(self) -> None:
        """Load index binary and metadata from disk."""
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded persistent FAISS index ({self.index.ntotal} vectors) from {self.index_path}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}. Re-initializing empty index.")
            self._create_empty_index()

    def delete_document(self, document_id: int, all_remaining_chunks: List[Dict[str, Any]] = None, all_remaining_embeddings: np.ndarray = None) -> None:
        """
        Delete document vectors from index by rebuilding with remaining chunks.
        """
        if all_remaining_chunks is not None and all_remaining_embeddings is not None:
            self.rebuild(all_remaining_chunks, all_remaining_embeddings)
        else:
            # Filter metadata in memory
            kept_meta = [m for m in self.metadata if m.get("document_id") != document_id]
            if len(kept_meta) == len(self.metadata):
                return
            # If no embeddings supplied, clear index if empty
            if not kept_meta:
                self._create_empty_index()
                self.save()
            else:
                logger.warning(f"Rebuilding index required to remove doc {document_id}")

    def rebuild(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Completely rebuild index from scratch with provided chunks and embeddings."""
        self._create_empty_index()
        if chunks and embeddings.size > 0:
            self.add_documents(chunks, embeddings)
        else:
            self.save()
