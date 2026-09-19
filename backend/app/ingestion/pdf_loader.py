import hashlib
import pymupdf as fitz
from pathlib import Path
from typing import List, Dict, Any
from app.core.logging import logger


class PDFExtractionError(Exception):
    """Exception raised when PDF parsing fails."""
    pass


class PDFLoader:
    """Page-aware PDF loader using PyMuPDF (fitz)."""

    @staticmethod
    def calculate_sha256(file_bytes: bytes) -> str:
        """Compute SHA-256 hash of PDF binary data for deduplication."""
        return hashlib.sha256(file_bytes).hexdigest()

    def extract_text_by_page(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text page-by-page from a PDF file.
        
        Returns:
            List of dicts: [
                {
                    "page_number": int (1-based),
                    "text": str,
                    "character_count": int
                }
            ]
        """
        path = Path(file_path)
        if not path.exists():
            raise PDFExtractionError(f"PDF file not found at: {file_path}")

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {file_path}: {e}")
            raise PDFExtractionError(f"Invalid or corrupted PDF file: {e}")

        pages_content = []
        try:
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                raw_text = page.get_text("text") or ""
                cleaned_text = self._clean_page_text(raw_text)

                pages_content.append({
                    "page_number": page_index + 1,
                    "text": cleaned_text,
                    "character_count": len(cleaned_text)
                })
        finally:
            doc.close()

        logger.info(f"Successfully extracted {len(pages_content)} pages from {path.name}")
        return pages_content

    @staticmethod
    def _clean_page_text(text: str) -> str:
        """Clean extracted page text removing excessive whitespace and null bytes."""
        if not text:
            return ""
        # Remove null bytes
        text = text.replace("\x00", "")
        # Normalize line endings & multiple spaces
        lines = [line.strip() for line in text.splitlines()]
        # Keep empty lines as single paragraph break if multiple exist
        cleaned_lines = []
        for line in lines:
            if line or (cleaned_lines and cleaned_lines[-1] != ""):
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()
