import logging
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Iterable

from jinja2 import Template
from pydantic import BaseModel

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
    TypedTool,
)

logger = logging.getLogger(__name__)


class Environment[T: BaseModel](ABC):
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
    async def get_tools(self) -> Iterable[TypedTool[type[T]]]:
        """Return the list of tools available to the agent in this environment."""
        raise NotImplementedError()


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


class DefaultEnvironment[T: BaseModel](Environment[T]):
    """
    Default environment implementation providing standard agent behavior.

    This class manages the agent's context, tool execution, and conversation flow, including:
    - Maintaining and summarizing conversation history
    - Rendering system prompts with available tools and instructions
    - Handling tool execution and termination logic
    - Supporting both synchronous and asynchronous tool lists
    - Integrating with OpenAI client configuration if provided
    """

    def __init__(
        self,
        tools: list[TypedTool[type[T]]]
        | Callable[[], Awaitable[list[TypedTool[type[T]]]]],
        extra_instructions: Callable[[], str] | str | None = None,
        history: History | None = None,
        max_history_length: int | None = settings.max_history_length,
        reduce_history_by: int = settings.reduce_history_by,
        openai_client: OpenAiClientConfig | None = None,
    ):
        """
        Initialize a DefaultEnvironment instance.

        Args:
            tools: List of available tools, or a callable returning a list of tools (can be async).
            extra_instructions: Optional string or callable providing additional instructions for the system prompt.
            history: Optional conversation history. If not provided, a new empty history is created.
            max_history_length: Maximum number of history entries to keep before summarizing.
            reduce_history_by: Number of entries to reduce when summarizing history.
            openai_client: Optional OpenAI client configuration for summarization.
        """
        self.extra_instructions = extra_instructions
        self.system_prompt_template: Template = Template(DEFAULT_SYSTEM_PROMPT_TEMPLATE)
        self.continuation_template: Template = Template(CONTINUATION_TEMPLATE)
        self.history = history or History.model_validate([])
        self.max_history_length = max_history_length
        self.reduce_history_by = reduce_history_by
        self.openai_client = openai_client
        self.provided_tools = tools

    async def get_context(self, remaining_iterations: int) -> History:
        """
        Build and return the conversation context for the agent.

        This method:
            - Summarizes history if it exceeds the maximum length
            - Removes any previous system instruction from the history
            - Renders and prepends a new system prompt with the current tools, remaining iterations, and extra instructions

        Args:
            remaining_iterations: Number of iterations left for the agent to complete its task

        Returns:
            Updated History object with the new system prompt prepended
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
                and "wingmate" in tool.meta
                and "TERMINATING" in tool.meta["wingmate"]
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
        history.add_message(
            role="system",
            content=system_prompt,
            flags=[MessageFlag.is_system_instruction],
            index=0,
        )
        logger.debug("History:\n" + str(history.compact()))

        return history

    async def call_tool(self, action: CallToolRequestParams) -> str | None:
        """
        Execute a tool action. (To be implemented by subclasses.)

        Args:
            action: The tool action to execute (parameters and tool name).

        Returns:
            The result of the tool execution as a string, or None to indicate termination.

        Raises:
            NotImplementedError: This base implementation must be overridden in a subclass.
        """
        raise NotImplementedError(
            "DefaultEnvironment does not implement tool calling. "
            "Subclass and override call_tool() to provide tool execution logic."
        )

    async def on_agent_message_completed(
        self, last_response: AgentResponse
    ) -> Message | TERMINATE:
        """
        Handle the agent's response after each message is completed.

        This method determines whether to terminate the conversation, execute a tool, or prompt the agent to continue:
            - If no tool action was taken, prompts the agent to continue or terminate
            - If a terminating tool was called, returns TERMINATE
            - Otherwise, executes the requested tool and returns the result as a Message

        Args:
            last_response: The most recent agent response to evaluate

        Returns:
            TERMINATE if a terminating tool was called
            Message with tool result if a tool was executed
            Message prompting continuation if no action was taken
        """
        terminating_tools = [
            tool.name
            for tool in await self.get_tools()
            if (
                tool.meta
                and "wingmate" in tool.meta
                and "TERMINATING" in tool.meta["wingmate"]
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

    async def get_tools(self) -> list[TypedTool[type[T]]]:
        """
        Return the list of tools available to the agent in this environment.

        Returns:
            List of BaseTool objects (may be resolved from a callable or returned directly)
        """
        return (
            await self.provided_tools()
            if callable(self.provided_tools)
            else self.provided_tools
        )
