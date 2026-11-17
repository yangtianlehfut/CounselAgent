"""LangGraph workflow for the legal consultation agent.

Case analysis and case search share a case-detail completion subgraph. The
adapter around that subgraph passes only the conversation messages and
recognized intent, keeping the parent's other state private. Ordinary
conversation uses a tool-free chat agent and bypasses all legal workflows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agent.agents import (
    IntentDecision,
    get_default_chat_agent,
    get_intent_agent,
    get_law_query_agent,
)
from src.agent.case_detail import (
    CASE_DETAIL_OUTPUT_KEYS,
    Intent,
    case_detail_subgraph,
)
from src.agent.prompts import LEGAL_AI_DISCLAIMER
from src.agent.tools import LAW_QUERY_TOOLS
from src.logger import get_logger


logger = get_logger(__name__)


class AgentState(TypedDict, total=False):
    """Shared state passed between workflow nodes."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    intent: Intent

    case_description: str
    case_results: list[dict[str, Any]]

    cause_of_action: str
    core_factors: list[str]
    missing_factors: list[str]
    user_answers: dict[str, str]
    synthesis: str

    final_response: str


NodeHandler = Callable[[AgentState], dict[str, Any]]


def _message_content_to_text(content: Any) -> str:
    """Extract text from a LangChain message content value."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _latest_ai_message_text(messages: list[AnyMessage]) -> str | None:
    """Return the latest final AI response text."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = _message_content_to_text(message.content).strip()
            if text:
                return text
    return None


def recognize_intent_node(state: AgentState) -> dict[str, Any]:
    """Recognize the legal workflow or ordinary conversation."""
    agent = get_intent_agent()
    result = agent.invoke({"messages": state.get("messages", [])})
    if isinstance(result, dict):
        decision = result.get("structured_response")
        if isinstance(decision, IntentDecision):
            logger.info("Intent recognized: %s", decision.intent)
            return {"intent": decision.intent}
    logger.warning(
        "Intent agent returned an unexpected result; falling back to default"
    )
    return {"intent": "default"}


def default_chat_node(state: AgentState) -> dict[str, Any]:
    """Reply through a tool-free agent without using any workflow branch."""
    messages = state.get("messages", [])
    result = get_default_chat_agent().invoke({"messages": messages})

    reply = next(
        (
            message
            for message in reversed(result.get("messages", []))
            if isinstance(message, AIMessage)
        ),
        None,
    )
    if reply is None:
        raise RuntimeError("Default chat agent returned no AI message")

    logger.info("Default chat response generated")
    return {
        "messages": [reply],
        "final_response": _message_content_to_text(reply.content),
    }


def law_query_node(state: AgentState) -> dict[str, Any]:
    """Answer a law query using an agent restricted to law-search tools."""
    messages = state.get("messages", [])
    result = get_law_query_agent(tools=LAW_QUERY_TOOLS).invoke(
        {"messages": messages}
    )

    result_messages = result.get("messages", [])
    new_messages = result_messages[len(messages) :]
    if not new_messages:
        raise RuntimeError("Law query agent returned no new messages")

    logger.info("Law query agent response generated")
    return {"messages": new_messages}


def case_search_node(state: AgentState) -> dict[str, Any]:
    """Answer a case query using an agent restricted to case-search tools."""
    # TODO: Bind a model that can call only the case-search tool. It may answer
    # directly without a tool call, then populate state["case_results"] or a
    # direct response. Never expose law-search tools in this branch.
    return {}


def comprehensive_analysis_node(state: AgentState) -> dict[str, Any]:
    """Combine statutes, precedents, facts, and answers into an analysis."""
    # TODO: Bind a model with both law-search and case-search tools, produce a
    # grounded legal analysis, and populate state["synthesis"].
    return {}


def output_node(state: AgentState) -> dict[str, Any]:
    model_response = _latest_ai_message_text(state.get("messages", []))
    if not model_response:
        raise RuntimeError("Output node received no model response")

    final_response = (
        f"{model_response.rstrip()}\n\n---\n\n{LEGAL_AI_DISCLAIMER}"
    )
    logger.info("Final response generated")
    return {"final_response": final_response}


def _route_by_intent(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == "default":
        destination = "default_chat"
    elif intent == "law_query":
        destination = "law_query"
    elif intent in {"case_search", "case_analysis"}:
        destination = "case_detail"
    else:
        destination = "default_chat"

    logger.info("Routing intent %s to node %s", intent, destination)
    return destination


def _route_after_case_detail(state: AgentState) -> str:
    """Resume the feature that invoked the shared detail subgraph."""
    if state.get("intent") == "case_search":
        return "case_search"
    return "comprehensive_analysis"


def complete_case_details_node(state: AgentState) -> dict[str, Any]:
    """Invoke the subgraph with only conversation messages and intent."""
    messages = state.get("messages", [])
    intent: Intent = state.get("intent", "case_analysis")

    result = case_detail_subgraph.invoke(
        {
            "messages": messages,
            "intent": intent,
        }
    )

    update = {
        key: result[key]
        for key in CASE_DETAIL_OUTPUT_KEYS
        if key in result
    }

    # The subgraph is append-only for messages. Return only newly added
    # messages so the parent's add_messages reducer does not re-add old ones.
    result_messages = result.get("messages", messages)
    new_messages = result_messages[len(messages) :]
    if new_messages:
        update["messages"] = new_messages

    return update


def build_graph(
    checkpointer: Any = None,
    node_overrides: Mapping[str, NodeHandler] | None = None,
):
    """Build and compile the legal consultation state graph.

    ``node_overrides`` is a build-time dependency injection hook used by tests
    and the debug entry point. Production nodes contain no mock logic.
    """
    workflow = StateGraph(AgentState)

    node_handlers: dict[str, NodeHandler] = {
        "recognize_intent": recognize_intent_node,
        "default_chat": default_chat_node,
        "law_query": law_query_node,
        "case_detail": complete_case_details_node,
        "case_search": case_search_node,
        "comprehensive_analysis": comprehensive_analysis_node,
        "output": output_node,
    }
    if node_overrides:
        node_handlers.update(node_overrides)

    for name, handler in node_handlers.items():
        workflow.add_node(name, handler)

    workflow.add_edge(START, "recognize_intent")

    workflow.add_conditional_edges("recognize_intent", _route_by_intent)

    workflow.add_conditional_edges("case_detail", _route_after_case_detail)

    workflow.add_edge("default_chat", "output")
    workflow.add_edge("law_query", "output")
    workflow.add_edge("case_search", "output")
    workflow.add_edge("comprehensive_analysis", "output")
    workflow.add_edge("output", END)

    return workflow.compile(checkpointer=checkpointer)


