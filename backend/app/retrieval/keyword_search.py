import re
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.db.models import Chunk, Document
from app.core.logging import logger, log_execution_time


class KeywordSearchRetriever:
    """
    Database-backed SQL keyword retriever.
    Effective for exact terms, policy numbers (e.g. HR-204), codes, dates, and acronyms.
    """

    def __init__(self, db: Session):
        self.db = db

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Execute SQL keyword search over chunks using term overlap and match frequency scoring.
        """
        if not query or not query.strip():
            return []

        with log_execution_time(f"Keyword Search ('{query}', top_k={top_k})"):
            # Extract query tokens (ignoring short stopwords)
            tokens = self._tokenize(query)
            if not tokens:
                tokens = [query.strip().lower()]

            # Build SQL OR conditions for tokens
            conditions = [Chunk.text.ilike(f"%{token}%") for token in tokens]
            
            query_results = (
                self.db.query(Chunk, Document.filename)
                .join(Document, Chunk.document_id == Document.id)
                .filter(or_(*conditions))
                .all()
            )

            scored_results = []
            for chunk, filename in query_results:
                score = self._compute_keyword_score(chunk.text, tokens, query)
                if score > 0:
                    scored_results.append({
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "document_name": filename,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "token_count": chunk.token_count,
                        "keyword_score": float(score)
                    })

            # Sort by keyword match score descending
            scored_results.sort(key=lambda x: x["keyword_score"], reverse=True)
            top_results = scored_results[:top_k]

            # Assign keyword rank (1-based)
            for rank, item in enumerate(top_results, start=1):
                item["keyword_rank"] = rank
                item["score"] = item["keyword_score"]

        logger.info(f"Keyword search returned {len(top_results)} matches for query '{query}'")
        return top_results

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize query string into search terms."""
        # Match words, hyphenated codes (e.g. HR-204), and alphanumerics
        tokens = re.findall(r"\b[A-Za-z0-9\-]+\b", text.lower())
        stopwords = {"a", "an", "the", "in", "on", "of", "and", "is", "for", "to", "what", "how", "who", "where", "which"}
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def _compute_keyword_score(self, text: str, tokens: List[str], full_query: str) -> float:
        """
        Compute term-frequency and exact phrase match score.
        Exact phrase match gives high boost.
        """
        text_lower = text.lower()
        score = 0.0

        # Exact phrase match boost
        if full_query.strip().lower() in text_lower:
            score += 5.0

        # Token match frequency
        for token in tokens:
            count = text_lower.count(token)
            if count > 0:
                # Diminishing marginal score for repeated terms
                score += (1.0 + (count - 1) * 0.2)

        return score
