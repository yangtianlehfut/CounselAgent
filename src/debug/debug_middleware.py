import json
from typing import Callable, Any

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from src.logger import get_logger

logger = get_logger(__name__)

@wrap_tool_call
def print_tool_call_args(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
    """Log every tool call's arguments and response."""
    tool_call = request.tool_call
    tool_name = tool_call["name"]
    tool_call_id = tool_call.get("id")

    logger.info(
        "Tool call arguments | name=%s | id=%s\n%s",
        tool_name,
        tool_call_id,
        _format_tool_log(tool_call.get("args", {})),
    )
    try:
        response = handler(request)
    except Exception:
        logger.exception(
            "Tool call failed | name=%s | id=%s",
            tool_name,
            tool_call_id,
        )
        raise

    logger.info(
        "Tool call response | name=%s | id=%s\n%s",
        tool_name,
        tool_call_id,
        _format_tool_log(response),
    )
    return response

def _format_tool_log(value: Any) -> str:
    """Serialize a tool argument or response for readable logging."""
    if isinstance(value, ToolMessage):
        value = value.model_dump()
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)