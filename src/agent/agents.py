"""Scenario-specific agent factories for the legal consultation workflow."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest, wrap_tool_call
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field, create_model

from src.agent.case_detail import Intent
from src.agent.prompts import (
    CAUSE_ANALYSIS_SYSTEM_PROMPT,
    CASE_ANALYSIS_SYSTEM_PROMPT,
    CASE_SEARCH_SYSTEM_PROMPT,
    DEFAULT_CHAT_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
    LAW_QUERY_SYSTEM_PROMPT,
)
from src.llm import LLMScenario, get_chat_model
from src.logger import get_logger
from src.retriever.crime_db_retriever import crime_db_retriever


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


OTHER_CAUSE_OF_ACTION = "其他"


def _format_supported_causes(crimes: Sequence[dict[str, Any]]) -> str:
    """Render supported causes and their constituent factors for the prompt."""
    blocks: list[str] = []
    for crime in crimes:
        name = crime.get("name") or ""
        article = crime.get("article") or ""
        header = f"### {name}（{article}）" if article else f"### {name}"
        lines = [header]
        constitution = (crime.get("constitution") or "").strip()
        non_constitution = (crime.get("non_constitution") or "").strip()
        if constitution:
            lines.append(f"- 构成：{constitution}")
        if non_constitution:
            lines.append(f"- 不构成：{non_constitution}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _build_cause_analysis_schema(crime_names: Sequence[str]) -> type[BaseModel]:
    """Build the decision schema, constraining cause names via Literal."""
    allowed_cause = Literal[(*crime_names, OTHER_CAUSE_OF_ACTION)]
    return create_model(
        "CauseAnalysisDecision",
        cause_name=(
            allowed_cause,
            Field(description="当前支持的案由名称；都不匹配时填“其他”"),
        ),
        most_likely_cause=(
            str,
            Field(
                default="",
                description="未命中支持范围时，你认为本案最可能的规范案由名称；命中时留空",
            ),
        ),
        reason=(str, Field(description="认定依据或未命中原因，不超过40字")),
    )


@lru_cache(maxsize=1)
def get_cause_analysis_agent():
    """Return the structured-output agent used for cause recognition."""
    crimes = crime_db_retriever.get_all_crime()
    system_prompt = CAUSE_ANALYSIS_SYSTEM_PROMPT.format(
        supported_causes=_format_supported_causes(crimes)
    )
    return create_agent(
        model=get_chat_model(LLMScenario.CAUSE_ANALYSIS),
        system_prompt=system_prompt,
        response_format=ToolStrategy(
            schema=_build_cause_analysis_schema(
                [crime["name"] for crime in crimes]
            )
        ),
        name="cause_analysis_agent",
    )


@lru_cache(maxsize=1)
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

@lru_cache(maxsize=1)
def get_case_search_agent(tools: Sequence[Any] = ()):
    """Build an agent that may only receive case-search tools."""
    return create_agent(
        model=get_chat_model(LLMScenario.CASE_SEARCH),
        tools=list(tools),
        system_prompt=CASE_SEARCH_SYSTEM_PROMPT,
        name="case_search_agent",
    )

@lru_cache(maxsize=1)
def get_case_analysis_agent(tools: Sequence[Any] = ()):
    """Build an agent that may receive law and case retrieval tools."""
    return create_agent(
        model=get_chat_model(LLMScenario.CASE_ANALYSIS),
        tools=list(tools),
        system_prompt=CASE_ANALYSIS_SYSTEM_PROMPT,
        name="case_analysis_agent",
    )
