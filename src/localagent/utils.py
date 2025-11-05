from typing import Annotated

from mcp.types import Tool
from pydantic import create_model
from pydantic.json_schema import WithJsonSchema

from .types import AgentResponse, CallToolRequestParams


def ignore_format_in_schema(input_schema: dict | list) -> dict | list:
    """Recursively remove 'format' keys from a JSON schema dictionary."""
    if isinstance(input_schema, dict):
        return {
            key: ignore_format_in_schema(value)
            for key, value in input_schema.items()
            if key != "format"
        }
    elif isinstance(input_schema, list):
        return [ignore_format_in_schema(item) for item in input_schema]
    else:
        return input_schema


def create_tool_call_schema(tool_name: str, input_schema: dict | list) -> dict | list:
    """Create a tool call schema from the given input schema."""
    return ignore_format_in_schema({
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "const": tool_name,
            },
            "arguments": input_schema,
        },
        "required": ["tool_name", "arguments"],
    })


def align_schema_with_tools[T: type[AgentResponse]](schema: T, tools: list[Tool]) -> T:
    # Convert model_fields to create_model format
    fields = {}
    for field_name, field_info in schema.model_fields.items():
        annotation = field_info.annotation

        metadata = list(field_info.metadata) if hasattr(field_info, "metadata") else []
        if field_name == "action":
            metadata.append(
                WithJsonSchema({
                    "anyOf": [
                        create_tool_call_schema(tool.name, tool.inputSchema)
                        for tool in tools
                    ]
                    + [{"type": "null"}]
                })
            )
            annotation = Annotated[CallToolRequestParams | None, *metadata]

        if metadata:
            annotation = Annotated[annotation, *metadata]

        fields[field_name] = (annotation, field_info)

    return create_model(schema.__name__, **fields)  # type: ignore
