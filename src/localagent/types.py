import logging
from enum import Enum
from typing import Any, Literal, Self, overload

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Sentinel

logger = logging.getLogger(__name__)

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
    id: SkipJsonSchema[int | None] = None
    role: Literal["system", "user", "assistant"]
    content: str
    flags: SkipJsonSchema[list[str]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class History(RootModel[tuple[Message, ...]]):
    root: tuple[Message, ...]

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        """Validate and ensure message IDs are correct after initialization."""
        return self.ensure_valid_ids()

    def ensure_valid_ids(self, raise_warning: bool = True) -> Self:
        """Ensure that all messages have valid IDs."""
        for idx, message in enumerate(self.root):
            if raise_warning and message.id is not None and message.id != idx:
                logger.warning(
                    f"Message ID {message.id} was incorrect.\nMessage: {message}"
                )
            message.id = idx
        return self

    def compact(self) -> list[dict[Literal["role", "content"], str]]:
        """Dump the history with minimal fields."""
        return [{"role": msg.role, "content": msg.content} for msg in self.root]

    @overload
    def add_message(self, msg: Message, *, index: int | None = None) -> None: ...

    @overload
    def add_message(
        self,
        *,
        role: Literal["system", "user", "assistant"],
        content: str,
        flags: list[str] | None = None,
        index: int | None = None,
    ) -> None: ...

    def add_message(
        self,
        msg: Message | None = None,
        *,
        role: Literal["system", "user", "assistant"] = "user",
        content: str = "",
        flags: list[str] | None = None,
        index: int | None = None,
    ) -> None:
        """Add a message to the history."""
        if index is None:
            index = len(self.root)
        if not (0 <= index <= len(self.root)):
            raise IndexError(
                f"Index out of bounds for adding message to history of length {len(self.root)}."
            )
        if msg is None:
            msg = Message(
                id=index,
                role=role,
                content=content,
                flags=flags or [],
            )
        else:
            msg.id = index
        self.root = self.root[:index] + (msg,) + self.root[index:]
        self.ensure_valid_ids(raise_warning=False)


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
