import re
from typing import List, Dict, Any


def evaluate_faithfulness(answer: str, context_chunks: List[Dict[str, Any]]) -> float:
    """
    Evaluate faithfulness: percentage of claim words in generated answer supported by context.
    Returns score between 0.0 and 1.0.
    """
    if not answer or not context_chunks:
        return 0.0

    if "could not find sufficient information" in answer.lower():
        # Correct refusal when context is missing is considered faithful
        return 1.0

    # Combine context text
    context_text = " ".join([c.get("text", "") for c in context_chunks]).lower()
    context_words = set(re.findall(r"\w+", context_text))

    answer_words = [
        w for w in re.findall(r"\w+", answer.lower())
        if w not in {"the", "a", "an", "is", "are", "and", "or", "to", "in", "on", "of", "for", "with", "by", "page"}
    ]

    if not answer_words:
        return 1.0

    supported_count = sum(1 for w in answer_words if w in context_words)
    return round(supported_count / len(answer_words), 4)


def evaluate_answer_relevance(question: str, answer: str) -> float:
    """
    Evaluate answer relevance: lexical overlap between query key terms and answer.
    Returns score between 0.0 and 1.0.
    """
    if not question or not answer:
        return 0.0

    query_keywords = set(
        w for w in re.findall(r"\w+", question.lower())
        if w not in {"what", "is", "the", "a", "an", "in", "on", "of", "and", "for", "to", "how", "who", "where", "which", "are"}
    )

    if not query_keywords:
        return 1.0

    answer_text = answer.lower()
    matched = sum(1 for kw in query_keywords if kw in answer_text)

    return round(matched / len(query_keywords), 4)
