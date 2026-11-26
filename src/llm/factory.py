"""Cached model factory used by business code."""

from __future__ import annotations

import sys
from enum import StrEnum
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from chromadb import EmbeddingFunction
from chromadb.api import Embeddable
from src.config import config
from src.llm.providers import (
    BaseLLMProvider,
    DeepSeekProvider,
    DeepSeekNonThinkingProvider,
    LocalVLLMProvider,
)


class LLMScenario(StrEnum):
    """Business scenarios that use a fixed LLM provider."""

    DEFAULT = "default"
    INTENT = "intent"
    LAW_QUERY = "law_query"
    CASE_SEARCH = "case_search"
    CASE_ANALYSIS = "case_analysis"
    SUMMARIZATION = "summarization"
    CAUSE_ANALYSIS = "cause_analysis"


SCENARIO_PROVIDERS: dict[LLMScenario, type[BaseLLMProvider]] = {
    LLMScenario.DEFAULT: DeepSeekProvider,
    LLMScenario.INTENT: DeepSeekNonThinkingProvider,
    LLMScenario.LAW_QUERY: DeepSeekProvider,
    LLMScenario.CASE_SEARCH: LocalVLLMProvider,
    LLMScenario.CASE_ANALYSIS: LocalVLLMProvider,
    LLMScenario.SUMMARIZATION: LocalVLLMProvider,
    LLMScenario.CAUSE_ANALYSIS: DeepSeekNonThinkingProvider,
}


@lru_cache(maxsize=None)
def _get_provider_llm(
    provider_type: type[BaseLLMProvider],
) -> BaseChatModel:
    """Create one cached LLM for each provider type."""
    return provider_type().get_llm()


def get_chat_model(scenario: LLMScenario) -> BaseChatModel:
    """Return the LLM assigned to the requested scenario."""
    try:
        provider_type = SCENARIO_PROVIDERS[scenario]
    except KeyError as exc:
        available = ", ".join(item.value for item in SCENARIO_PROVIDERS)
        raise ValueError(
            f"Unknown LLM scenario {scenario!r}; expected one of: {available}"
        ) from exc
    return _get_provider_llm(provider_type)

def get_embedding_function() -> EmbeddingFunction[Embeddable]:
    if config.EMBEDDING_PROVIDER != "ollama":
        sys.exit(f"暂不支持的 EMBEDDING_PROVIDER：{config.EMBEDDING_PROVIDER}（目前只支持 ollama）")

    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

    return OllamaEmbeddingFunction(model_name=config.EMBEDDING_MODEL_NAME)
