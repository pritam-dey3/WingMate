from abc import ABC, abstractmethod

from jinja2 import Template
from mcp.types import Tool

from .types import AgentResponse, CallToolRequestParams, History, Message, MessageFlag


class Environment(ABC):
    """Defines the environment in which the agent operates, including tools, context, and termination conditions."""

    system_prompt_template: Template
    tools: list[Tool]

    @abstractmethod
    async def get_context(self, history: History, remaining_iterations: int) -> History:
        """Modify and return the conversation context based on history and remaining iterations."""
        pass

    @abstractmethod
    async def on_agent_message_completed(
        self, last_response: AgentResponse
    ) -> Message | None:
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
        pass


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
- {{ tool.name }}: {{ tool.description }}
  Input Schema: {{ tool.inputSchema }}
{% endfor %}

When you decide to use a tool, provide the tool name and arguments in your response. After the tool call, you will receive the result which you should use to continue the conversation.

You must return a valid json object as per the schema provided.

Example response:
{
  "thought": "I need to use a tool",
  "msg_to_user": "<some message to keep user engaged>", (message to user should not contain any error, tool call or technical information)
  "action": {
    "name": "tool_name",
    "arguments": {
      "arg1": "value1",
      "arg2": "value2"
    }
  }
}

You must not make a lot of tool calls and try to respond with the answer as early as possible ({{ remaining_iterations }} iterations remaining).
You must respond with the `answer` tool when you have the final answer.

{% if extra_instructions %}
{{ extra_instructions }}
{% endif %}
"""


class DefaultEnvironment(Environment):
    """Default environment implementation with standard behavior."""

    def __init__(self, tools: list[Tool], extra_instructions: str | None = None):
        """
        Initialize the default environment.

        Args:
            tools: List of tools available to the agent (answer and follow_up tools are added automatically).
            extra_instructions: Additional instructions to append to the system prompt.
        """
        self.extra_instructions = extra_instructions
        self.system_prompt_template = Template(DEFAULT_SYSTEM_PROMPT_TEMPLATE)
        self.tools = list(tools) + [answer_tool, follow_up_tool]

    async def get_context(self, history: History, remaining_iterations: int) -> History:
        """
        Build context by prepending system prompt to history.

        Removes any existing system instruction and prepends a new one with
        current tools and remaining iterations.
        """
        history = history.model_copy(deep=True)

        # Remove existing system instruction if present
        if history.root and MessageFlag.is_system_instruction in history.root[0].flags:
            history.root = history.root[1:]

        # Filter tools if filter function is provided
        tools_to_show = self.tools

        # Render system prompt
        system_prompt = self.system_prompt_template.render(
            tools=tools_to_show,
            remaining_iterations=remaining_iterations,
            extra_instructions=self.extra_instructions,
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

        return new_history

    async def call_tool(self, action: CallToolRequestParams) -> str:
        """
        Default implementation raises NotImplementedError.

        Subclasses must override this to provide actual tool execution.
        """
        raise NotImplementedError(
            "DefaultEnvironment does not implement tool calling. "
            "Subclass and override call_tool() or use a concrete environment implementation."
        )

    async def on_agent_message_completed(
        self, last_response: AgentResponse
    ) -> Message | None:
        """
        Default implementation that handles termination, tool execution, and continuation logic.

        Returns:
            None if agent should terminate (answer/follow_up tools used).
            Message with tool result if a tool was called.
            Message with error if no action was taken.
        """
        # Error if no action was taken
        if not last_response.action:
            return Message(
                role="user",
                content="No action taken. You must conclude with an answer or a query.",
                flags=[MessageFlag.is_system_response],
            )
        elif last_response.action.name in ["answer", "follow_up"]:
            return None

        # Execute the tool and return the result as a Message
        tool_result = await self.call_tool(last_response.action)
        return Message(
            role="user",
            content=f"Tool result: {tool_result}",
            flags=[MessageFlag.is_tool_result],
        )
