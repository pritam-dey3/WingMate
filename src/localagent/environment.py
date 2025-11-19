import logging
from abc import ABC, abstractmethod
from typing import Awaitable, Callable

from jinja2 import Template
from mcp.types import Tool

from .history_utils import create_summary_entry, last_summary_index
from .settings import settings
from .types import (
    TERMINATE,
    AgentResponse,
    CallToolRequestParams,
    History,
    Message,
    MessageFlag,
    OpenAiClientConfig,
)

logger = logging.getLogger(__name__)


class Environment(ABC):
    """Defines the environment in which the agent operates, including tools, context, and termination conditions."""

    history: History

    @abstractmethod
    async def get_context(self, remaining_iterations: int) -> History:
        """Modify and return the conversation context based on history and remaining iterations."""
        raise NotImplementedError()

    @abstractmethod
    async def on_agent_message_completed(
        self, last_response: AgentResponse
    ) -> Message | TERMINATE:
        """Hook called after each agent message is completed.

        This method should:
        1. Perform any side effects (logging, printing, etc.)
        2. Decide whether to continue or terminate the agent's turn
        3. If continuing, execute any tool calls and return a Message to add to history

        Args:
            last_response: The most recent agent response to evaluate.

        Returns:
            None to terminate the agent's turn.
            A Message object to continue the conversation (with appropriate role, content, and flags).
        """
        raise NotImplementedError()

    @abstractmethod
    async def get_tools(self) -> list[Tool]:
        """Return the list of tools available to the agent in this environment."""
        raise NotImplementedError()


# Terminating action tools
answer_tool = Tool(
    name="answer",
    description="Provide the final answer to the user's query based on the information gathered.",
    inputSchema={
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The final answer to the user's query.",
            },
        },
        "required": ["answer"],
    },
)

follow_up_tool = Tool(
    name="follow_up",
    description="Ask a follow-up question to gather more information from the user.",
    inputSchema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The follow-up question to ask the user.",
            },
        },
        "required": ["question"],
    },
)


DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI agent with access to the following tools:

{% for tool in tools %}
- {{ tool.name }}: {{ tool.description | replace('\n', '\n    ') }}
    Input Schema: {{ tool.inputSchema | safe }}
{% endfor %}

When you decide to use a tool, provide the tool name and arguments in your response. After the tool call, you will receive the result which you should use to continue the conversation.

You must return a valid json object as per the schema provided.

Example response:
{
    "msg_to_user": "<some message to keep user engaged>", (message to user should not contain any error, tool call or technical information)
    "action": {
        "name": "tool_name",
        "arguments": {
            "arg1": "value1",
            "arg2": "value2"
        }
    }
}

{% if remaining_iterations <= 7 %}
CRITICAL: You have {{ remaining_iterations }} iterations remaining. Minimize tool calls.
{% else %}
CRITICAL: Try to complete your task in as few iterations as possible.
{% endif %}

{% if terminating_tools|length != 0 %}
TERMINATION REQUIREMENT: As soon as you have sufficient information to answer the user's query, you MUST immediately call one of these terminating tools: {{ terminating_tools | map(attribute='name') | join(', ') }}. Do NOT make additional tool calls after you can answer. DO NOT continue exploring unnecessarily.
{% endif %}

{% if extra_instructions %}
{{ extra_instructions }}

Complete the task and IMMEDIATELY end conversation with the appropriate terminating tool once done. Do NOT wait or ask for confirmation.
{% endif %}
"""

CONTINUATION_TEMPLATE = """No tool was called in the last response. If you have not yet reached a conclusion, please feel free to explore.
{% if terminating_tools|length != 0 %}
If you have, you must conclude by calling any one of the terminating tool(s): {{ terminating_tools | join(', ') }}.
{% endif %}"""


class DefaultEnvironment(Environment):
    """Default environment implementation with standard behavior."""

    def __init__(
        self,
        tools: list[Tool] | Callable[[], Awaitable[list[Tool]]],
        extra_instructions: Callable[[], str] | str | None = None,
        history: History | None = None,
        max_history_length: int | None = settings.max_history_length,
        reduce_history_by: int = settings.reduce_history_by,
        openai_client: OpenAiClientConfig | None = None,
    ):
        """
        Initialize the default environment.

        Args:
            tools: List of tools available to the agent (answer and follow_up tools are added automatically).
            extra_instructions: Additional instructions to append to the system prompt.
        """
        self.extra_instructions = extra_instructions
        self.system_prompt_template: Template = Template(DEFAULT_SYSTEM_PROMPT_TEMPLATE)
        self.continuation_template: Template = Template(CONTINUATION_TEMPLATE)
        self.history = history or History.model_validate([])
        self.max_history_length = max_history_length
        self.reduce_history_by = reduce_history_by
        self.openai_client = openai_client
        self.tools = tools

    async def get_context(self, remaining_iterations: int) -> History:
        """
        Build context by prepending system prompt to history.

        Removes any existing system instruction and prepends a new one with
        current tools and remaining iterations.
        """

        if (
            self.max_history_length is not None
            and len(self.history.root) - last_summary_index(self.history)
            > self.max_history_length
        ):
            self.history = await create_summary_entry(
                old_history=self.history,
                reduce_by=self.reduce_history_by,
                client_config=self.openai_client,
            )

        # Extract relevant history after last summary
        history = History.model_validate(
            self.history.root[last_summary_index(self.history) :]
        )
        if history.root and MessageFlag.is_system_instruction in history.root[0].flags:
            history.root = history.root[1:]

        # Render system prompt
        tools = await self.get_tools()
        terminating_tools = [
            tool
            for tool in tools
            if (
                tool.meta
                and "localagent" in tool.meta
                and "TERMINATING" in tool.meta["localagent"]
            )
        ]
        system_prompt = self.system_prompt_template.render(
            tools=tools,
            remaining_iterations=remaining_iterations,
            extra_instructions=self.extra_instructions()
            if callable(self.extra_instructions)
            else self.extra_instructions,
            terminating_tools=terminating_tools,
        )

        # Prepend system prompt to history
        new_history = History.model_validate(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                    "flags": [MessageFlag.is_system_instruction],
                }
            ]
            + history.root
        )
        logger.debug("History:\n" + new_history.model_dump_json(indent=2))

        return new_history

    async def call_tool(self, action: CallToolRequestParams) -> str | None:
        """
        Default implementation raises NotImplementedError.

        Subclasses must override this to provide actual tool execution.
        Args:
            action: The tool action to execute.
        Returns:
            The result of the tool execution. If the tool returns `None`, it indicates termination.
        """
        raise NotImplementedError(
            "DefaultEnvironment does not implement tool calling. "
            "Subclass and override call_tool() or use a concrete environment implementation."
        )

    async def on_agent_message_completed(
        self, last_response: AgentResponse
    ) -> Message | TERMINATE:
        """
        Default implementation that handles termination, tool execution, and continuation logic.

        Returns:
            None if agent should terminate (answer/follow_up tools used).
            Message with tool result if a tool was called.
            Message with error if no action was taken.
        """
        terminating_tools = [
            tool.name
            for tool in await self.get_tools()
            if (
                tool.meta
                and "localagent" in tool.meta
                and "TERMINATING" in tool.meta["localagent"]
            )
        ]
        # Error if no action was taken
        if not last_response.action:
            return Message(
                role="user",
                content=self.continuation_template.render(
                    terminating_tools=terminating_tools
                ),
                flags=[MessageFlag.is_system_response],
            )

        if len(terminating_tools) == 0:
            logger.warning("No terminating tools found in environment.")
        if last_response.action.tool_name in terminating_tools:
            return TERMINATE

        # Execute the tool and return the result as a Message
        tool_result = await self.call_tool(last_response.action)
        return Message(
            role="user",
            content=f"Tool result: {tool_result}",
            flags=[MessageFlag.is_tool_result],
        )

    async def get_tools(self) -> list[Tool]:
        """Return the list of tools available to the agent in this environment."""
        return await self.tools() if callable(self.tools) else self.tools
