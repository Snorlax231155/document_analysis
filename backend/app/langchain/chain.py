from typing import List, Dict, Any, Tuple
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document

from app.langchain.documents import convert_chunks_to_langchain_documents, format_langchain_documents_for_context
from app.langchain.prompts import get_langchain_rag_prompt
from app.langchain.llm import LangChainLLMAdapter
from app.langchain.output_parser import get_langchain_output_parser
from app.core.logging import logger, log_execution_time


class LangChainRAGChain:
    """
    LangChain LCEL (LangChain Expression Language) Runnable Pipeline.
    Orchestrates prompt construction, LLM invocation via adapter, and string output parsing.
    """

    def __init__(self, llm_adapter: LangChainLLMAdapter = None):
        self.llm_adapter = llm_adapter or LangChainLLMAdapter()
        self.prompt_template = get_langchain_rag_prompt()
        self.output_parser = get_langchain_output_parser()
        
        # Build modern LCEL Runnable Pipeline
        self.chain = (
            RunnablePassthrough()
            | self.prompt_template
            | self.llm_adapter
            | self.output_parser
        )

    def run(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, bool, List[Document]]:
        """
        Execute LangChain LCEL pipeline.
        
        Returns:
            Tuple[raw_answer_text, answer_found_boolean, langchain_documents_list]
        """
        # 1. Convert native retrieved chunks to LangChain Document objects
        langchain_docs = convert_chunks_to_langchain_documents(retrieved_chunks)

        # 2. Format context for prompt
        context_str = format_langchain_documents_for_context(langchain_docs)

        # 3. Invoke LCEL Runnable Chain
        with log_execution_time("LangChain LCEL Pipeline Execution"):
            answer_text = self.chain.invoke({
                "context": context_str,
                "question": question
            })

        answer_str = str(answer_text).strip()
        refusal_phrase = "I could not find sufficient information in the provided documents"
        answer_found = refusal_phrase.lower() not in answer_str.lower()

        return answer_str, answer_found, langchain_docs
