from typing import Literal, Union

from json_schema_to_pydantic import create_model as create_model_from_json_schema
from mcp.types import Tool
from pydantic import Field, create_model

from .types import AgentResponse, CallToolRequestParams


def align_schema_with_tools[T: type[AgentResponse]](schema: T, tools: list[Tool]) -> T:
    actions = []
    for tool in tools:
        model = create_model(
            tool.name + "InputModel",
            tool_name=(Literal[tool.name], Field(default=tool.name)),
            arguments=(create_model_from_json_schema(tool.inputSchema), Field(...)),
            __base__=CallToolRequestParams,
        )
        actions.append(model)

    return create_model(
        schema.__name__,
        action=(Union[tuple(actions) + (type(None),)], Field(default=None)),
        __base__=schema,
    )
