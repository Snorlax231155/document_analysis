from typing import List, Dict, Any

SYSTEM_PROMPT = """You are an Enterprise Document Intelligence & RAG Assistant.

Your task is to answer the user's question based STRICTLY and ONLY on the supplied context chunks extracted from enterprise documents.

CRITICAL INSTRUCTIONS:
1. Use ONLY the provided document context below to answer the question.
2. Do NOT invent facts, assume information, or extrapolate beyond the provided text.
3. If the context does NOT contain sufficient information to answer the question, respond EXACTLY with:
   "I could not find sufficient information in the provided documents to answer this question."
4. Cite the source document name and page number for key facts mentioned in your answer (e.g., "[employee_handbook.pdf, Page 14]").
5. If different documents conflict, clearly state the discrepancy and cite both sources.
6. Answer concisely, professionally, and clearly.
"""


def build_rag_prompt(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Format RAG prompt combining question and context chunks with metadata.
    """
    if not context_chunks:
        formatted_context = "No relevant document context found."
    else:
        context_blocks = []
        for idx, chunk in enumerate(context_chunks, start=1):
            doc_name = chunk.get("document_name", "Unknown Document")
            page_num = chunk.get("page_number", "?")
            text = chunk.get("text", "").strip()
            context_blocks.append(
                f"--- CONTEXT BLOCK {idx} [Document: {doc_name} | Page: {page_num}] ---\n{text}"
            )
        formatted_context = "\n\n".join(context_blocks)

    user_prompt = f"""Context:
{formatted_context}

Question:
{question}

Answer:"""
    return user_prompt
