import logging
from enum import Enum
from typing import Any, Generic, Literal, Self, TypeVar, overload

from json_schema_to_pydantic import create_model as json_schema_to_pydantic_model
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from pydantic._internal._model_construction import ModelMetaclass
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Sentinel

logger = logging.getLogger(__name__)

TERMINATE = Sentinel("TERMINATE")


class McpToolAnnotations(BaseModel):
    title: str | None = None
    readOnlyHint: bool | None = None
    destructiveHint: bool | None = None
    idempotentHint: bool | None = None
    openWorldHint: bool | None = None
    model_config = ConfigDict(extra="allow")


class McpTool(BaseModel):
    """MCP Tool object."""

    name: str
    title: str | None = None
    description: str | None = None
    inputSchema: dict[str, Any]
    outputSchema: dict[str, Any] | None = None
    icons: list | None = None
    annotations: McpToolAnnotations | None = None
    meta: dict[str, Any] | None = Field(alias="_meta", default=None)
    model_config = ConfigDict(extra="allow")


class CallToolRequestParams[T: BaseModel](BaseModel):
    tool_name: str
    arguments: T


class WingmateError(Exception):
    pass


class MaxAgentIterationsExceededError(WingmateError):
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


class AgentResponse[T: BaseModel](BaseModel):
    """Response for the given user query, including the agent's thought process"""

    msg_to_user: str | None = None
    action: CallToolRequestParams[T] | None = None
    turn_completed: SkipJsonSchema[bool] = False


class AgentResponseThoughtful[T: BaseModel](AgentResponse[T]):
    """Response for the given user query"""

    thought: str | None = None


class Token(BaseModel):
    token: str


class OpenAiClientConfig(BaseModel):
    llm_model_name: str
    base_url: str
    api_key: str | None
    extra_kw: dict = {}


T_co = TypeVar("T_co", bound=type[BaseModel], covariant=True)


class TypedTool(McpTool, Generic[T_co]):
    model_config = ConfigDict(frozen=True)
    input_model: SkipJsonSchema[T_co] = Field(exclude=True)

    @model_validator(mode="before")
    @classmethod
    def validate_input_schema(cls, data: Any) -> Any:
        if isinstance(data, dict):
            input_model = data.get("input_model")
            input_schema = data.get("inputSchema")

            if input_model is None and input_schema is None:
                raise ValueError("Either input_model or inputSchema must be provided.")
            elif input_model is not None:
                if input_schema is not None:
                    logger.warning(
                        "Ignoring inputSchema since input_model is provided."
                    )
                data["inputSchema"] = input_model.model_json_schema()
            elif input_model is None:
                assert input_schema is not None
                data["input_model"] = json_schema_to_pydantic_model(
                    input_schema, base_model_type=BaseToolModel
                )

            if (
                "meta" in data
                and data["meta"]
                and ("wingmate" in data["meta"])
                and ("TERMINATING" in data["meta"]["wingmate"])
            ):
                data["input_model"].__is_terminating__ = True

        return data

    @classmethod
    def from_tool(cls, tool: McpTool):
        return cls.model_validate(tool.model_dump())


class BaseToolMeta(ModelMetaclass):
    __is_terminating__: bool = False

    def __new__(mcls, name, bases, attrs, **kwargs):
        terminating = kwargs.pop("terminating", False)
        cls = super().__new__(mcls, name, bases, attrs, **kwargs)
        if terminating:
            cls.__is_terminating__ = True
        return cls


class BaseToolModel(BaseModel, metaclass=BaseToolMeta):
    @classmethod
    def convert_to_tool(cls) -> TypedTool[type[Self]]:
        if cls.__is_terminating__:
            meta = {"wingmate": ["TERMINATING"]}
        else:
            meta = {}
        tool = TypedTool(
            name=cls.__name__,
            description=cls.__doc__ or "",
            input_model=cls,
            _meta=meta,
        )  # type: ignore
        return tool


class TextStream(BaseModel):
    type: Literal["text"] = "text"
    delta: str


class ToolCallStream[T: BaseModel](BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call: CallToolRequestParams[T]
