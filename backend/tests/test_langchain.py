import pytest
from langchain_core.documents import Document
from app.langchain.documents import convert_chunks_to_langchain_documents, format_langchain_documents_for_context
from app.langchain.prompts import get_langchain_rag_prompt
from app.langchain.llm import LangChainLLMAdapter
from app.langchain.output_parser import get_langchain_output_parser
from app.langchain.chain import LangChainRAGChain


def test_chunk_to_langchain_document_conversion():
    native_chunks = [
        {
            "chunk_id": 101,
            "document_id": 5,
            "document_name": "employee_handbook.pdf",
            "page_number": 14,
            "chunk_index": 2,
            "text": "Employees are entitled to 20 days of annual leave.",
            "rrf_score": 0.032,
            "rerank_score": 0.89
        }
    ]

    docs = convert_chunks_to_langchain_documents(native_chunks)
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert doc.page_content == "Employees are entitled to 20 days of annual leave."
    assert doc.metadata["document_name"] == "employee_handbook.pdf"
    assert doc.metadata["page_number"] == 14
    assert doc.metadata["chunk_id"] == 101


def test_langchain_document_formatting():
    docs = [
        Document(page_content="Leave policy text.", metadata={"document_name": "hr.pdf", "page_number": 3})
    ]
    formatted = format_langchain_documents_for_context(docs)
    assert "--- CONTEXT BLOCK 1 [Document: hr.pdf | Page: 3] ---" in formatted
    assert "Leave policy text." in formatted


def test_langchain_prompt_template():
    prompt = get_langchain_rag_prompt()
    formatted_msg = prompt.format_messages(context="Sample context", question="Sample question")
    assert len(formatted_msg) == 2
    assert "Sample context" in formatted_msg[1].content
    assert "Sample question" in formatted_msg[1].content


def test_langchain_lcel_chain_execution():
    chain = LangChainRAGChain()
    native_chunks = [
        {
            "chunk_id": 1,
            "document_id": 1,
            "document_name": "security.pdf",
            "page_number": 4,
            "chunk_index": 0,
            "text": "Remote work security policy is designated under code SEC-402.",
            "rerank_score": 0.95
        }
    ]

    answer_text, answer_found, langchain_docs = chain.run(
        question="What is the remote work security code?",
        retrieved_chunks=native_chunks
    )

    assert isinstance(answer_text, str)
    assert answer_found is True
    assert len(langchain_docs) == 1
    assert langchain_docs[0].metadata["document_name"] == "security.pdf"
    assert langchain_docs[0].metadata["page_number"] == 4
