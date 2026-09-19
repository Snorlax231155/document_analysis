from typing import List, Dict, Any
from langchain_core.documents import Document


def convert_chunks_to_langchain_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert native retrieved chunk dictionaries into LangChain Document objects,
    preserving full document and page citation metadata.
    """
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk.get("text", "").strip(),
            metadata={
                "document_id": chunk.get("document_id", 0),
                "document_name": chunk.get("document_name", "Unknown Document"),
                "page_number": chunk.get("page_number", 1),
                "chunk_id": chunk.get("chunk_id", 0),
                "chunk_index": chunk.get("chunk_index", 0),
                "token_count": chunk.get("token_count", 0),
                "rrf_score": chunk.get("rrf_score", 0.0),
                "rerank_score": chunk.get("rerank_score", chunk.get("score", 0.0))
            }
        )
        documents.append(doc)
    return documents


def format_langchain_documents_for_context(documents: List[Document]) -> str:
    """
    Format a list of LangChain Document objects into a clean, context string for LLM prompts.
    """
    if not documents:
        return "No relevant document context found."

    context_blocks = []
    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata
        doc_name = meta.get("document_name", "Unknown Document")
        page_num = meta.get("page_number", "?")
        context_blocks.append(
            f"--- CONTEXT BLOCK {idx} [Document: {doc_name} | Page: {page_num}] ---\n{doc.page_content}"
        )
    return "\n\n".join(context_blocks)
