from langchain_core.output_parsers import StrOutputParser


def get_langchain_output_parser() -> StrOutputParser:
    """
    Return standard LangChain StrOutputParser for LCEL Runnable pipeline execution.
    """
    return StrOutputParser()
