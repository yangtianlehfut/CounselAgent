"""Tools available to the legal consultation agents."""

from src.agent.tools.law_query import (
    LAW_QUERY_TOOLS,
    get_all_laws,
    get_articles,
)

__all__ = ["LAW_QUERY_TOOLS", "get_all_laws", "get_articles"]
