import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.db.database import get_db
from app.db.repository import DocumentRepository
from app.ingestion.pdf_loader import PDFLoader, PDFExtractionError
from app.ingestion.chunker import DocumentChunker
from app.embeddings.embedder import EmbeddingService
from app.retrieval.vector_store import FAISSVectorStore
from app.schemas.documents import DocumentResponseSchema, DocumentDetailSchema

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentResponseSchema, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a PDF document.
    Performs SHA-256 deduplication, page-aware text extraction, chunking, vector indexing, and SQL storage.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported."
        )

    repo = DocumentRepository(db)
    loader = PDFLoader()
    chunker = DocumentChunker()
    embedder = EmbeddingService()
    vector_store = FAISSVectorStore()

    # Read binary bytes for hash deduplication
    file_bytes = file.file.read()
    file.file.seek(0)

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    file_hash = loader.calculate_sha256(file_bytes)

    # Check duplicate document
    existing_doc = repo.get_by_hash(file_hash)
    if existing_doc:
        logger.info(f"Document duplicate detected: {file.filename} (Hash: {file_hash[:8]}...)")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document '{file.filename}' already uploaded (ID: {existing_doc.id})."
        )

    # Save PDF to disk
    save_filename = f"{file_hash[:12]}_{file.filename}"
    saved_path = settings.DOCUMENTS_DIR / save_filename
    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    try:
        # 1. Extract text page-by-page
        pages_content = loader.extract_text_by_page(str(saved_path))
        page_count = len(pages_content)

        if page_count == 0:
            saved_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No readable text found in PDF document."
            )

        # 2. Save initial Document DB record
        doc_record = repo.create_document(
            filename=file.filename,
            file_path=str(saved_path),
            file_hash=file_hash,
            file_size=len(file_bytes),
            page_count=page_count
        )

        # 3. Chunk page text while preserving page number & document metadata
        all_chunks_data = []
        global_chunk_idx = 0
        for page_data in pages_content:
            page_chunks = chunker.chunk_page_text(
                page_text=page_data["text"],
                page_number=page_data["page_number"],
                document_id=doc_record.id,
                document_name=doc_record.filename,
                start_chunk_index=global_chunk_idx
            )
            all_chunks_data.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        if not all_chunks_data:
            repo.delete_document(doc_record.id)
            saved_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="PDF content contained no chunkable text sentences."
            )

        # 4. Save chunks to SQLite DB
        chunk_records = repo.save_chunks(doc_record.id, all_chunks_data)

        # Build chunk list formatted for vector store
        faiss_chunks = []
        texts_to_embed = []
        for c_rec in chunk_records:
            texts_to_embed.append(c_rec.text)
            faiss_chunks.append({
                "chunk_id": c_rec.id,
                "document_id": doc_record.id,
                "document_name": doc_record.filename,
                "page_number": c_rec.page_number,
                "chunk_index": c_rec.chunk_index,
                "text": c_rec.text,
                "token_count": c_rec.token_count
            })

        # 5. Embed chunks using SentenceTransformers
        embeddings = embedder.embed_documents(texts_to_embed)

        # 6. Add embeddings & metadata to FAISS index
        vector_store.add_documents(chunks=faiss_chunks, embeddings=embeddings)

        logger.info(f"Successfully processed PDF '{file.filename}' (Doc ID {doc_record.id}, Chunks {len(chunk_records)})")
        return doc_record

    except PDFExtractionError as e:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error processing document upload: {e}")
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process PDF: {e}")


@router.get("", response_model=List[DocumentResponseSchema])
def list_documents(db: Session = Depends(get_db)):
    """List all uploaded PDF documents."""
    repo = DocumentRepository(db)
    return repo.list_all()


@router.get("/{document_id}", response_model=DocumentDetailSchema)
def get_document_detail(document_id: int, db: Session = Depends(get_db)):
    """Get document details along with extracted text chunks."""
    repo = DocumentRepository(db)
    doc = repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document from SQL database, storage, and rebuild FAISS index."""
    repo = DocumentRepository(db)
    doc = repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    # Remove physical file
    try:
        Path(doc.file_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Could not remove physical file {doc.file_path}: {e}")

    # Delete from DB
    repo.delete_document(document_id)

    # Rebuild FAISS vector store with remaining chunks
    remaining_chunks = repo.get_all_chunks()
    embedder = EmbeddingService()
    vector_store = FAISSVectorStore()

    if remaining_chunks:
        texts = [c["text"] for c in remaining_chunks]
        embeddings = embedder.embed_documents(texts)
        vector_store.rebuild(remaining_chunks, embeddings)
    else:
        vector_store.rebuild([], embedder.embed_documents([]))

    return {"message": f"Document ID {document_id} ('{doc.filename}') deleted successfully."}
