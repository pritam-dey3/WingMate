from .agent import LocalAgent
from .environment import DefaultEnvironment, Environment, answer_tool, follow_up_tool
from .types import (
    AgentResponse,
    CallToolRequestParams,
    History,
    LocalAgentError,
    MaxAgentIterationsExceededError,
    Message,
    MessageFlag,
    Token,
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
    "answer_tool",
    "follow_up_tool",
]


def main() -> None:
    print("Hello from localagent!")
