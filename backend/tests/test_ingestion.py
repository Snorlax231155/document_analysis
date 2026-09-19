import pytest
from app.ingestion.chunker import DocumentChunker
from app.ingestion.pdf_loader import PDFLoader


def test_chunker_basic_splitting():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "Employees are entitled to 20 days of paid annual leave per year. "
        "Leave requests must be submitted two weeks in advance. "
        "Unused leave carries over up to 5 days."
    )
    chunks = chunker.chunk_page_text(
        page_text=sample_text,
        page_number=1,
        document_id=10,
        document_name="handbook.pdf"
    )

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk["document_id"] == 10
        assert chunk["document_name"] == "handbook.pdf"
        assert chunk["page_number"] == 1
        assert "text" in chunk
        assert chunk["token_count"] > 0


def test_chunker_overlap():
    chunker = DocumentChunker(chunk_size=60, chunk_overlap=25)
    text = "Sentence one is brief. Sentence two is longer and explanatory. Sentence three wraps up."
    chunks = chunker.chunk_page_text(text, page_number=2, document_id=1, document_name="doc.pdf")

    assert len(chunks) > 1


def test_pdf_loader_sha256():
    loader = PDFLoader()
    data = b"%PDF-1.4 sample pdf binary data content"
    sha = loader.calculate_sha256(data)
    assert len(sha) == 64
    assert sha == loader.calculate_sha256(data)
