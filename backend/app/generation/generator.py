from typing import List, Dict, Any, Tuple
from app.generation.llm_provider import LLMProvider, get_llm_provider
from app.generation.prompts import build_rag_prompt, SYSTEM_PROMPT
from app.core.logging import logger, log_execution_time


class AnswerGenerator:
    """
    RAG Answer Generator managing prompt formatting, LLM provider invocation,
    answer grounding detection, and citation structuring.
    """

    REFUSAL_PHRASE = "I could not find sufficient information in the provided documents"

    def __init__(self, provider: LLMProvider = None):
        self.provider = provider or get_llm_provider()

    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, bool, List[Dict[str, Any]]]:
        """
        Generate answer from question and retrieved context chunks.
        
        Returns:
            Tuple[answer_text, answer_found_boolean, citations_list]
        """
        if not context_chunks:
            return (
                "I could not find sufficient information in the provided documents to answer this question.",
                False,
                []
            )

        prompt = build_rag_prompt(question, context_chunks)

        with log_execution_time("RAG Generation"):
            raw_answer = self.provider.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)

        answer_text = raw_answer.strip()
        answer_found = self.REFUSAL_PHRASE.lower() not in answer_text.lower()

        citations = []
        if answer_found:
            citations = self._extract_citations(context_chunks)

        return answer_text, answer_found, citations

    def _extract_citations(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate and structure source page citations from used context chunks."""
        seen = set()
        citations = []

        for chunk in context_chunks:
            doc_id = chunk.get("document_id", 0)
            doc_name = chunk.get("document_name", "Unknown Document")
            page_num = chunk.get("page_number", 1)
            chunk_id = chunk.get("chunk_id", 0)
            score = chunk.get("score", chunk.get("rerank_score", 0.0))

            key = (doc_id, page_num)
            if key not in seen:
                seen.add(key)
                citations.append({
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "page_number": page_num,
                    "chunk_id": chunk_id,
                    "score": round(score, 4)
                })

        return citations
