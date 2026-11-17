import os
from abc import abstractmethod, ABC
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


class BaseLLMProvider(ABC):
    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        pass


class LocalVLLMProvider(BaseLLMProvider):
    def get_llm(self) -> BaseChatModel:
        return ChatOpenAI(
            model='Qwen3.5-0.8B',
            base_url='http://localhost:8000/v1',
            api_key='empty',
            temperature=0.8,
            use_responses_api=False,
            max_tokens=1024,
        )

class DeepSeekProvider(BaseLLMProvider):
    def get_llm(self) -> BaseChatModel:
        return self._build_llm()

    def _build_llm(
        self,
        *,
        model_kwargs: dict[str, Any] | None = None,
    ) -> BaseChatModel:
        load_dotenv()
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise Exception('DEEPSEEK_API_KEY not set')

        extra_model_kwargs = (
            {"extra_body": model_kwargs}
            if model_kwargs is not None
            else {}
        )
        return ChatDeepSeek(
            model='deepseek-flash',
            temperature=0.8,
            max_tokens=1024,
            **extra_model_kwargs,
        )


class DeepSeekNonThinkingProvider(DeepSeekProvider):
    def get_llm(self) -> BaseChatModel:
        return self._build_llm(
            model_kwargs={"thinking": {"type": "disabled"}},
        )


class OllamaProvider(BaseLLMProvider):
    def get_llm(self) -> BaseChatModel:
        return ChatOllama(
            model='deepseek-r1:8b',
            temperature=0.8
        )
