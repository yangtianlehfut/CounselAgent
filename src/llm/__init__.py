"""Public interface for model resolution."""

from src.llm.factory import LLMScenario, get_chat_model

__all__ = [
    "LLMScenario",
    "get_chat_model",
]
