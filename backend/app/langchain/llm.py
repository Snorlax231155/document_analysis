from typing import List, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from app.generation.llm_provider import LLMProvider, get_llm_provider
from app.generation.prompts import SYSTEM_PROMPT


class LangChainLLMAdapter(BaseChatModel):
    """
    LangChain BaseChatModel adapter wrapping our native LLMProvider abstraction.
    Allows seamless execution in LCEL Runnable chains while maintaining support for OpenAI API and Mock LLM.
    """
    provider: Any = None

    def __init__(self, provider: Optional[LLMProvider] = None, **kwargs):
        super().__init__(**kwargs)
        self.provider = provider or get_llm_provider()

    @property
    def _llm_type(self) -> str:
        return "custom_enterprise_llm_adapter"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> ChatResult:
        # Extract system prompt and user query from LangChain message objects
        system_content = SYSTEM_PROMPT
        user_content = ""

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_content = str(msg.content)
            elif isinstance(msg, HumanMessage):
                user_content = str(msg.content)
            else:
                user_content += f"\n{msg.content}"

        raw_output = self.provider.generate(
            prompt=user_content,
            system_prompt=system_content
        )

        message = AIMessage(content=raw_output)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
