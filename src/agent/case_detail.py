"""Case-detail completion subgraph for the legal consultation agent."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


Intent = Literal["default", "law_query", "case_search", "case_analysis"]


class CaseDetailState(TypedDict, total=False):
    """State owned by the case-detail subgraph.

    The parent adapter supplies only ``messages`` and ``intent``. The remaining
    fields are outputs produced inside the subgraph.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    intent: Intent
    case_description: str
    cause_of_action: str
    core_factors: list[str]
    missing_factors: list[str]
    user_answers: dict[str, str]


def cause_of_action_node(state: CaseDetailState) -> dict[str, Any]:
    """Determine the legal cause of action."""
    # TODO: Infer the cause of action from the current case facts and
    # state["messages"], then populate state["cause_of_action"].
    return {}


def get_core_factors_node(state: CaseDetailState) -> dict[str, Any]:
    """Load the core factors required for the identified cause of action."""
    # TODO: Load the core factors for state["cause_of_action"] from the crime
    # database and populate state["core_factors"].
    return {}


def ask_missing_factors_node(state: CaseDetailState) -> dict[str, Any]:
    """Identify and ask for facts that are still missing."""
    # TODO: Compare known facts with state["core_factors"], populate
    # state["missing_factors"], ask the user for missing details, and collect
    # the answers in state["user_answers"].
    return {}


def build_case_detail_subgraph():
    """Build the reusable case-detail completion subgraph."""
    workflow = StateGraph(CaseDetailState)

    workflow.add_node("cause_of_action", cause_of_action_node)
    workflow.add_node("get_core_factors", get_core_factors_node)
    workflow.add_node("ask_missing_factors", ask_missing_factors_node)

    workflow.add_edge(START, "cause_of_action")
    workflow.add_edge("cause_of_action", "get_core_factors")
    workflow.add_edge("get_core_factors", "ask_missing_factors")
    workflow.add_edge("ask_missing_factors", END)

    return workflow.compile()


case_detail_subgraph = build_case_detail_subgraph()


CASE_DETAIL_OUTPUT_KEYS = (
    "case_description",
    "cause_of_action",
    "core_factors",
    "missing_factors",
    "user_answers",
)
