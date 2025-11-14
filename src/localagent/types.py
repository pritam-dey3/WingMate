from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Sentinel

TERMINATE = Sentinel("TERMINATE")


class CallToolRequestParams(BaseModel):
    tool_name: str
    arguments: Any


class LocalAgentError(Exception):
    pass


class MaxAgentIterationsExceededError(LocalAgentError):
    pass


class MessageFlag(str, Enum):
    is_system_instruction = "is_system_instruction"
    is_system_response = "is_system_response"
    is_tool_result = "is_tool_result"
    is_summary = "is_summary"


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    flags: SkipJsonSchema[list[str]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


History = RootModel[list[Message]]


class AgentResponse(BaseModel):
    """Response for the given user query, including the agent's thought process"""

    thought: str | None = None
    msg_to_user: str | None = None
    action: CallToolRequestParams | None = None
    turn_completed: SkipJsonSchema[bool] = False


class Token(BaseModel):
    token: str


class OpenAiClientConfig(BaseModel):
    llm_model_name: str
    base_url: str
    api_key: str | None
    extra_kw: dict = {}
