import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logging import logger, log_execution_time


class EmbeddingService:
    """
    Sentence Transformer embedding service with model caching and L2 vector normalization.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._model = None
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading SentenceTransformer embedding model: {settings.EMBEDDING_MODEL}")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self._model

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate L2 normalized embeddings for a list of text strings.
        Normalization enables cosine similarity via FAISS IndexFlatIP.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        with log_execution_time("Embedding Documents"):
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate L2 normalized embedding for a single query string."""
        with log_execution_time("Embedding Query"):
            embedding = self.model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        return embedding.astype(np.float32)
