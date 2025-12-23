from .agent import Agent
from .environment import DefaultEnvironment, Environment
from .types import (
    AgentResponse,
    BaseToolModel,
    CallToolRequestParams,
    History,
    MaxAgentIterationsExceededError,
    Message,
    MessageFlag,
    Token,
    TypedTool,
    WingmateError,
)

__all__ = [
    "AgentResponse",
    "CallToolRequestParams",
    "History",
    "WingmateError",
    "MaxAgentIterationsExceededError",
    "Message",
    "MessageFlag",
    "Token",
    # agent
    "Agent",
    # environment
    "Environment",
    "DefaultEnvironment",
    "TypedTool",
    "BaseToolModel",
]


def main() -> None:
    print("Hello from wingmate!")
