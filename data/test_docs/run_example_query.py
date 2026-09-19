import json
from pathlib import Path
from sqlalchemy.orm import Session
from app.db.database import SessionLocal, init_db
from app.db.repository import DocumentRepository
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.chunker import DocumentChunker
from app.embeddings.embedder import EmbeddingService
from app.retrieval.vector_store import FAISSVectorStore
from app.rag.pipeline import RAGPipeline

def run_example():
    init_db()
    db = SessionLocal()
    pdf_path = Path("data/test_docs/acme_enterprise_security_and_hr_policy_2026.pdf")
    
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.")
        return

    loader = PDFLoader()
    chunker = DocumentChunker()
    embedder = EmbeddingService()
    vector_store = FAISSVectorStore()
    repo = DocumentRepository(db)

    file_bytes = pdf_path.read_bytes()
    file_hash = loader.calculate_sha256(file_bytes)

    # Check if already processed
    existing_doc = repo.get_by_hash(file_hash)
    if not existing_doc:
        print(f"Ingesting document: {pdf_path.name}...")
        pages_content = loader.extract_text_by_page(str(pdf_path))
        
        doc_record = repo.create_document(
            filename=pdf_path.name,
            file_path=str(pdf_path),
            file_hash=file_hash,
            file_size=len(file_bytes),
            page_count=len(pages_content)
        )

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

        chunk_records = repo.save_chunks(doc_record.id, all_chunks_data)

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

        embeddings = embedder.embed_documents(texts_to_embed)
        vector_store.add_documents(chunks=faiss_chunks, embeddings=embeddings)
        print(f"Successfully processed {len(chunk_records)} chunks into FAISS index.")

    # Run sample RAG Query
    pipeline = RAGPipeline(db=db)
    question = "What are the remote work security requirements under policy SEC-402?"
    print(f"\nExecuting query: '{question}'...")
    response = pipeline.answer(question=question, top_k=5, enable_reranking=True)

    print("\n--- RAG PIPELINE RESPONSE JSON ---")
    print(json.dumps(response, indent=2))
    db.close()

if __name__ == "__main__":
    run_example()
