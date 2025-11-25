from typing import Iterable, Literal, Union, overload

from pydantic import BaseModel

from .types import AgentResponse, AgentResponseThoughtful, TypedTool


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
    if disable_thought:
        return AgentResponse[Union[*tuple(tool.input_model for tool in tools)]]
    else:
        return AgentResponseThoughtful[
            Union[*tuple(tool.input_model for tool in tools)]
        ]
