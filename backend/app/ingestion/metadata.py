from typing import Dict, Any


def format_chunk_metadata(
    chunk_id: int,
    document_id: int,
    document_name: str,
    page_number: int,
    chunk_index: int,
    text: str,
    token_count: int
) -> Dict[str, Any]:
    """
    Format standard chunk metadata payload stored alongside FAISS and search indexes.
    """
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "document_name": document_name,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "text": text,
        "token_count": token_count
    }
