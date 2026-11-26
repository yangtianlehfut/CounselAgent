"""Case-detail completion subgraph for the legal consultation agent."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from src.logger import get_logger
from src.retriever.crime_db_retriever import crime_db_retriever


logger = get_logger(__name__)

Intent = Literal["default", "law_query", "case_search", "case_analysis"]


class CaseDetailState(TypedDict, total=False):
    """State owned by the case-detail subgraph.

    The parent adapter supplies only ``messages`` and ``intent``. The remaining
    fields are outputs produced inside the subgraph.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    intent: Intent
    cause_name: str
    cause_supported: bool
    core_factors: list[str]
    missing_factors: list[str]
    user_answers: dict[str, str]


def cause_analysis_node(state: CaseDetailState) -> dict[str, Any]:
    """Determine the legal cause of action."""
    # Imported lazily to avoid a circular import: agents imports Intent from
    # this module at module load time.
    from src.agent.agents import OTHER_CAUSE_OF_ACTION, get_cause_analysis_agent

    messages = list(state.get("messages", []))
    if not messages:
        logger.warning("Cause-of-action node received no messages")
        return {}

    crime_names = [crime["name"] for crime in crime_db_retriever.get_all_crime()]
    result = get_cause_analysis_agent().invoke({"messages": messages})

    decision = result.get("structured_response") if isinstance(result, dict) else None
    if not isinstance(decision, BaseModel):
        logger.warning(
            "Cause-of-action agent returned an unexpected result: %r", result
        )
        return {}

    recognized = (decision.cause_name or "").strip()
    cause_name = (
        decision.most_likely_cause.strip()
        if recognized == OTHER_CAUSE_OF_ACTION
        else recognized
    )

    if cause_name in crime_names:
        logger.info(
            "Cause of action recognized: %s (reason: %s)",
            cause_name,
            decision.reason,
        )
        return {"cause_name": cause_name, "cause_supported": True}

    logger.info(
        "Cause of action unsupported; recognized=%r, most likely=%r (reason: %s)",
        recognized,
        cause_name,
        decision.reason,
    )
    return {"cause_name": cause_name, "cause_supported": False}


def get_core_factors_node(state: CaseDetailState) -> dict[str, Any]:
    """Load the core factors required for the identified cause of action."""
    # TODO: Load the core factors for state["cause_name"] from the crime
    # database and populate state["core_factors"].
    return {}


def ask_missing_factors_node(state: CaseDetailState) -> dict[str, Any]:
    """Identify and ask for facts that are still missing."""
    # TODO: Compare known facts with state["core_factors"], populate
    # state["missing_factors"], ask the user for missing details, and collect
    # the answers in state["user_answers"].
    return {}


def _route_after_cause_analysis(state: CaseDetailState) -> str:
    """Stop the subgraph when the cause of action is not supported."""
    if state.get("cause_supported"):
        return "get_core_factors"
    logger.info("Unsupported cause of action; skip the remaining subgraph nodes")
    return END


def build_case_detail_subgraph():
    """Build the reusable case-detail completion subgraph."""
    workflow = StateGraph(CaseDetailState)

    workflow.add_node("cause_analysis", cause_analysis_node)
    workflow.add_node("get_core_factors", get_core_factors_node)
    workflow.add_node("ask_missing_factors", ask_missing_factors_node)

    workflow.add_edge(START, "cause_analysis")
    workflow.add_conditional_edges(
        "cause_analysis",
        _route_after_cause_analysis,
        ["get_core_factors", END],
    )
    workflow.add_edge("get_core_factors", "ask_missing_factors")
    workflow.add_edge("ask_missing_factors", END)

    return workflow.compile()


case_detail_subgraph = build_case_detail_subgraph()


CASE_DETAIL_OUTPUT_KEYS = (
    "cause_name",
    "cause_supported",
    "user_answers",
)
