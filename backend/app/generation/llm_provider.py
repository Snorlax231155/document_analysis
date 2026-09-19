import re
import httpx
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.core.logging import logger, log_execution_time
from app.generation.prompts import SYSTEM_PROMPT, build_rag_prompt


class LLMProvider(ABC):
    """Abstract base class for provider-independent LLM generation."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.0
    ) -> str:
        """Generate text completion from prompt."""
        pass


class OpenAILLMProvider(LLMProvider):
    """OpenAI-compatible REST API client."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None
    ):
        self.api_key = api_key or settings.LLM_API_KEY
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self.model = model or settings.LLM_MODEL

    def generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.0
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }

        url = f"{self.base_url}/chat/completions"
        with log_execution_time(f"OpenAI LLM API Call ({self.model})"):
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.error(f"LLM API request failed: {e}")
                raise RuntimeError(f"LLM generation failed: {e}")


class MockLLMProvider(LLMProvider):
    """
    Offline Mock LLM engine.
    Extracts answers directly from retrieved context chunks when external API key is omitted.
    Refuses when context does not contain sufficient query overlap.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.0
    ) -> str:
        with log_execution_time("Mock LLM Generation"):
            # Parse prompt to extract Question and Context blocks
            q_match = re.search(r"Question:\s*(.*?)(?:\nAnswer:|$)", prompt, re.DOTALL)
            question = q_match.group(1).strip() if q_match else ""

            # Check if prompt contains context blocks
            if "No relevant document context found" in prompt or not question:
                return "I could not find sufficient information in the provided documents to answer this question."

            # Extract context text blocks
            blocks = re.findall(r"--- CONTEXT BLOCK \d+ \[Document: (.*?) \| Page: (\d+)\] ---\n(.*?)(?=\n--- CONTEXT BLOCK|\n\nQuestion:|$)", prompt, re.DOTALL)
            
            if not blocks:
                return "I could not find sufficient information in the provided documents to answer this question."

            query_words = set(re.findall(r"\w+", question.lower())) - {"what", "is", "the", "a", "an", "in", "on", "of", "and", "for", "to", "how", "who", "where", "which"}
            
            best_sentence = ""
            best_doc = ""
            best_page = ""
            best_score = 0

            for doc_name, page_num, text in blocks:
                sentences = re.split(r"(?<=[.!?])\s+", text)
                for sent in sentences:
                    sent_clean = sent.strip()
                    if len(sent_clean) < 10:
                        continue
                    sent_words = set(re.findall(r"\w+", sent_clean.lower()))
                    overlap = len(query_words.intersection(sent_words))
                    if overlap > best_score:
                        best_score = overlap
                        best_sentence = sent_clean
                        best_doc = doc_name
                        best_page = page_num

            if best_score == 0 or not best_sentence:
                return "I could not find sufficient information in the provided documents to answer this question."

            return f"Based on the provided documents: {best_sentence} [{best_doc}, Page {best_page}]"


class OllamaLLMProvider(LLMProvider):
    """Local Ollama REST API client."""

    def __init__(
        self,
        base_url: str = None,
        model: str = None
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL

    def generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.0
    ) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "options": {
                "temperature": temperature
            },
            "stream": False
        }

        with log_execution_time(f"Ollama Local LLM Call ({self.model})"):
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    if "message" in data and "content" in data["message"]:
                        return data["message"]["content"].strip()
                    elif "response" in data:
                        return data["response"].strip()
                    raise ValueError("Unexpected payload format from Ollama API.")
            except Exception as e:
                logger.error(f"Ollama request failed: {e}. Falling back to Mock LLM provider.")
                mock = MockLLMProvider()
                return mock.generate(prompt=prompt, system_prompt=system_prompt, temperature=temperature)


def get_llm_provider() -> LLMProvider:
    """
    Factory function returning:
    - OllamaLLMProvider if LLM_PROVIDER is 'ollama' or Ollama is running locally
    - OpenAILLMProvider if LLM_API_KEY is provided
    - MockLLMProvider as offline fallback
    """
    provider_setting = settings.LLM_PROVIDER.lower().strip()

    if provider_setting == "ollama":
        logger.info(f"Using OllamaLLMProvider (Model: {settings.OLLAMA_MODEL})")
        return OllamaLLMProvider()

    if settings.LLM_API_KEY and settings.LLM_API_KEY.strip():
        logger.info("Using OpenAILLMProvider")
        return OpenAILLMProvider()

    # Check if Ollama is accessible locally
    try:
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        with httpx.Client(timeout=1.5) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                if models:
                    logger.info(f"Detected running local Ollama service with models: {models}. Using OllamaLLMProvider ({models[0]})")
                    return OllamaLLMProvider(model=models[0])
    except Exception:
        pass

    logger.info("No external LLM key or Ollama service detected. Using MockLLMProvider for offline execution.")
    return MockLLMProvider()

