"""Scenario-specific agent factories for the legal consultation workflow."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest, wrap_tool_call
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel

from src.agent.case_detail import Intent
from src.agent.prompts import (
    CASE_ANALYSIS_SYSTEM_PROMPT,
    CASE_SEARCH_SYSTEM_PROMPT,
    DEFAULT_CHAT_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
    LAW_QUERY_SYSTEM_PROMPT,
)
from src.llm import LLMScenario, get_chat_model
from src.logger import get_logger


logger = get_logger(__name__)

class IntentDecision(BaseModel):
    intent: Intent


@lru_cache(maxsize=1)
def get_intent_agent():
    """Return the structured-output agent used for intent recognition."""
    return create_agent(
        model=get_chat_model(LLMScenario.INTENT),
        system_prompt=INTENT_SYSTEM_PROMPT,
        response_format=ToolStrategy(schema=IntentDecision),
        name="intent_agent",
    )


@lru_cache(maxsize=1)
def get_default_chat_agent():
    """Return the tool-free agent used for ordinary conversation."""
    return create_agent(
        model=get_chat_model(LLMScenario.DEFAULT),
        tools=[],
        system_prompt=DEFAULT_CHAT_SYSTEM_PROMPT,
        name="default_chat_agent",
    )


def get_law_query_agent(tools: Sequence[Any] = ()):
    """Build an agent that may only receive law-query tools."""
    middleware = []
    try:
        from src.debug.debug_middleware import print_tool_call_args
        middleware.append(print_tool_call_args)
    except ImportError:
        pass

    return create_agent(
        model=get_chat_model(LLMScenario.LAW_QUERY),
        tools=list(tools),
        system_prompt=LAW_QUERY_SYSTEM_PROMPT,
        name="law_query_agent",
        middleware=middleware,
    )


def get_case_search_agent(tools: Sequence[Any] = ()):
    """Build an agent that may only receive case-search tools."""
    return create_agent(
        model=get_chat_model(LLMScenario.CASE_SEARCH),
        tools=list(tools),
        system_prompt=CASE_SEARCH_SYSTEM_PROMPT,
        name="case_search_agent",
    )


def get_case_analysis_agent(tools: Sequence[Any] = ()):
    """Build an agent that may receive law and case retrieval tools."""
    return create_agent(
        model=get_chat_model(LLMScenario.CASE_ANALYSIS),
        tools=list(tools),
        system_prompt=CASE_ANALYSIS_SYSTEM_PROMPT,
        name="case_analysis_agent",
    )
