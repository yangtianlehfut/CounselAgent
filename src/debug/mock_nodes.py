"""Centralized node mocks for interactive workflow debugging.

This package is not imported by the production workflow. Delete ``src/debug``
to remove all debug mocks when they are no longer needed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, get_args

from src.agent.workflow import AgentState


MockNode = Callable[[AgentState], dict[str, Any]]

def mock_recognize_intent(_state: AgentState) -> dict[str, Any]:
    return {"intent": 'law_query'}


_MOCK_NODES: dict[str, MockNode] = {
    "recognize_intent": mock_recognize_intent,
}


def get_mock_node_overrides() -> dict[str, MockNode]:
    return _MOCK_NODES
