from .agent import LocalAgent
from .environment import DefaultEnvironment, Environment
from .types import (
    AgentResponse,
    BaseToolModel,
    CallToolRequestParams,
    History,
    LocalAgentError,
    MaxAgentIterationsExceededError,
    Message,
    MessageFlag,
    Token,
    TypedTool,
)

__all__ = [
    "AgentResponse",
    "CallToolRequestParams",
    "History",
    "LocalAgentError",
    "MaxAgentIterationsExceededError",
    "Message",
    "MessageFlag",
    "Token",
    # agent
    "LocalAgent",
    # environment
    "Environment",
    "DefaultEnvironment",
    "TypedTool",
    "BaseToolModel",
]


def main() -> None:
    print("Hello from localagent!")
