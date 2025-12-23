from typing import Iterable, Literal, Union, overload

from fastmcp import Client
from pydantic import BaseModel

from .types import AgentResponse, AgentResponseThoughtful, BaseToolModel, TypedTool


@overload
def build_agent_response_schema[T: BaseModel](
    disable_thought: Literal[True], tools: Iterable[TypedTool[type[T]]]
) -> type[AgentResponse[T]]: ...
@overload
def build_agent_response_schema[T: BaseModel](
    disable_thought: Literal[False], tools: Iterable[TypedTool[type[T]]]
) -> type[AgentResponseThoughtful[T]]: ...
def build_agent_response_schema[T: BaseModel](
    disable_thought: bool, tools: Iterable[TypedTool[type[T]]]
) -> type[AgentResponse[T] | AgentResponseThoughtful[T]]:
    if len(list(tools)) == 0:
        tool_type = None
    else:
        tool_type = Union[*tuple(tool.input_model for tool in tools)]
    if disable_thought:
        return AgentResponse[tool_type]  # type: ignore
    else:
        return AgentResponseThoughtful[tool_type]  # type: ignore


def mcp_tools(client: Client):
    """
    Returns a callable that, when invoked, asynchronously retrieves and returns a list of
    TypedTool instances corresponding to the tools available from the given fastmcp Client.

    Args:
        client (Client): An instance of fastmcp.Client used to list available tools.

    Returns:
        Callable[[], Coroutine[None, None, list[TypedTool[type[BaseToolModel]]]]]:
            An async function that returns a list of TypedTool objects when awaited.
    """

    async def tools() -> list[TypedTool[type[BaseToolModel]]]:
        async with client:
            return [TypedTool.from_tool(tool) for tool in await client.list_tools()]

    return tools
