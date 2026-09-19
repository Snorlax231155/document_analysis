import re
from typing import List, Dict, Any
from app.core.config import settings
from app.core.logging import logger


class DocumentChunker:
    """
    Intelligent token & sentence-aware document chunker.
    Preserves page boundaries and paragraph context.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def chunk_page_text(
        self,
        page_text: str,
        page_number: int,
        document_id: int,
        document_name: str,
        start_chunk_index: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Splits page text into configurable size chunks with overlap,
        respecting sentence boundaries where possible.
        """
        if not page_text or not page_text.strip():
            return []

        # Split text into sentences/paragraphs
        sentences = self._split_into_sentences(page_text)
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        chunk_idx = start_chunk_index

        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If a single sentence exceeds chunk size, split it by characters
            if sentence_len > self.chunk_size:
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences).strip()
                    chunks.append(self._create_chunk_dict(
                        chunk_text, chunk_idx, page_number, document_id, document_name
                    ))
                    chunk_idx += 1
                    current_chunk_sentences = []
                    current_length = 0

                # Split long sentence
                sub_chunks = self._slice_long_text(sentence)
                for sub in sub_chunks:
                    chunks.append(self._create_chunk_dict(
                        sub, chunk_idx, page_number, document_id, document_name
                    ))
                    chunk_idx += 1
                continue

            if current_length + sentence_len + 1 > self.chunk_size and current_chunk_sentences:
                # Store current accumulated chunk
                chunk_text = " ".join(current_chunk_sentences).strip()
                chunks.append(self._create_chunk_dict(
                    chunk_text, chunk_idx, page_number, document_id, document_name
                ))
                chunk_idx += 1

                # Calculate overlap sentences
                overlap_sentences = []
                overlap_len = 0
                for prev_sent in reversed(current_chunk_sentences):
                    if overlap_len + len(prev_sent) + 1 <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_len += len(prev_sent) + 1
                    else:
                        break

                current_chunk_sentences = overlap_sentences + [sentence]
                current_length = overlap_len + sentence_len + 1
            else:
                current_chunk_sentences.append(sentence)
                current_length += sentence_len + 1

        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            chunks.append(self._create_chunk_dict(
                chunk_text, chunk_idx, page_number, document_id, document_name
            ))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences preserving sentence terminators."""
        pattern = r"(?<=[.!?])\s+"
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p.strip()]

    def _slice_long_text(self, text: str) -> List[str]:
        """Fallback hard slice for unusually long blocks of text without punctuation."""
        slices = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(text), step):
            slices.append(text[i : i + self.chunk_size])
        return slices

    def _create_chunk_dict(
        self,
        text: str,
        chunk_index: int,
        page_number: int,
        document_id: int,
        document_name: str
    ) -> Dict[str, Any]:
        """Construct standard chunk dictionary with page metadata."""
        # Estimate token count (~4 characters per token average)
        token_count = max(1, len(text) // 4)
        return {
            "chunk_index": chunk_index,
            "document_id": document_id,
            "document_name": document_name,
            "page_number": page_number,
            "text": text,
            "token_count": token_count,
            "character_count": len(text)
        }
