from collections.abc import AsyncGenerator, Sequence
from typing import Awaitable, Callable

from mcp.types import Tool

from .llm import stream_agent_response
from .settings import settings
from .types import (
    AgentResponse,
    CallToolRequestParams,
    History,
    MaxAgentIterationsExceededError,
    Message,
    MessageFlag,
)

# terminating actions
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


class LocalAgent:
    """
    An agent that orchestrates LLM interactions with tool calling capabilities.

    The agent follows a loop pattern:
    1. Update history with user query/tool results
    2. Context engineering (with optional callback)
    3. Stream LLM response
    4. Tool call (with optional callback)
    5. Either terminate turn or continue loop with tool results

    ---
    config:
    theme: redux
    ---
    flowchart TD
        Q["user query"] --> L["loop"]
        L --> H[Update History]
        H --> CE(("Context<br>Engineering<br>hook"))
        CE --> S["Stream"]
        S --> RS[Full Response]
        RS -.- RSH[LLM Response<br>hook]
        RS --> TC(("Tool Call"))
        TC --> TCR["Result"]
        TCR --> TT[Terminate]
        TCR --> L

        L@{ shape: terminal}
        TT@{ shape: terminal}
    """

    def __init__(
        self,
        tools: Sequence[Tool],
        agent_message_completion_hook: Callable[
            [History, AgentResponse], Awaitable[None]
        ]
        | None = None,
        context_engineering_hook: Callable[
            [History, Sequence[Tool], int], Awaitable[History]
        ]
        | None = None,
        tool_call_callback: Callable[[CallToolRequestParams], Awaitable[str]]
        | None = None,
        max_iterations: int = settings.max_agent_iterations,
        message_separation_token: str = "\n\n",
    ):
        """
        Initialize the SimpleAgent.

        Args:
            tools: Sequence of MCP tools available to the agent. The agent automatically
                adds 'answer' and 'follow_up' tools for terminating actions.
            agent_message_completion_hook: Optional async callback invoked after each agent
                message is completed. Takes the conversation history and the agent's response,
                returns None.
            context_engineering_hook: Optional async callback for modifying the conversation
                context before sending to the LLM. Takes the current history, available tools,
                and remaining iterations, returns modified history.
            tool_call_callback: Optional async callback for executing tool calls. Takes a
                CallToolRequestParams object containing the tool name and arguments,
                returns the tool execution result as a string.
            max_iterations: Maximum number of agent loop iterations to prevent infinite loops.
                Defaults to settings.max_agent_iterations.
            message_separation_token: Token used to separate messages when streaming responses.
                Defaults to "\n\n".
        """
        self.tools = list(tools) + [answer_tool, follow_up_tool]
        self.agent_message_completion_hook = agent_message_completion_hook
        self.context_engineering_hook = context_engineering_hook
        self.tool_call_callback = tool_call_callback
        self.max_iterations = max_iterations
        self.message_separation_token = message_separation_token

    async def run(self, history: History):
        """
        Run the agent loop for a given user query.

        This method implements the agent's main loop:
        - Applies context engineering
        - Streams LLM responses
        - Handles tool calls
        - Continues loop until termination

        Args:
            history: The conversation history to process

        Yields:
            Message objects as they are generated during the agent's execution

        Raises:
            MaxAgentIterationsExceededError: If the agent exceeds max_iterations
        """
        history = history.model_copy(deep=True)

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1

            # Context Engineering Hook
            if self.context_engineering_hook:
                context = await self.context_engineering_hook(
                    history, tuple(self.tools), self.max_iterations - iteration
                )
            else:
                context = history

            response: AgentResponse | None = None
            async for response in stream_agent_response(context, AgentResponse):
                yield response

            # Agent Message Completion Hook
            if response:
                history.root.append(
                    Message(
                        role="assistant",
                        content=response.model_dump_json(indent=2),
                    )
                )
                if self.agent_message_completion_hook:
                    await self.agent_message_completion_hook(history, response)

            if response and response.action and self.tool_call_callback:
                if response.action.name in ["answer", "follow_up"]:
                    return
                tool_call_result = await self.tool_call_callback(response.action)

                # Add tool result to history and continue loop
                history.root.append(
                    Message(
                        role="user",
                        content=f"Tool result: {tool_call_result}",
                        flags=[MessageFlag.is_tool_result],
                    )
                )
            else:
                history.root.append(
                    Message(
                        role="user",
                        content="No action taken. You must conclude with an answer or a query.",
                        flags=[MessageFlag.is_system_response],
                    )
                )

        raise MaxAgentIterationsExceededError(
            f"Agent exceeded maximum iterations ({self.max_iterations})"
        )

    async def stream(self, history: History) -> AsyncGenerator[str, None]:
        """
        Stream the agent's responses as plain text for a given user query.

        This method wraps the run() method and yields only the new/incremental
        text responses as they arrive.

        Args:
            history: The conversation history to process

        Yields:
            Incremental plain text responses from the agent (only new content)

        Raises:
            MaxAgentIterationsExceededError: If the agent exceeds max_iterations
        """
        prev_content = ""
        async for response in self.run(history):
            if not response.msg_to_user:
                continue

            if response.msg_to_user.startswith(prev_content):
                new_content = response.msg_to_user[len(prev_content) :]
                if new_content:
                    prev_content = response.msg_to_user
                    yield new_content
            else:
                yield self.message_separation_token
                prev_content = response.msg_to_user
                yield response.msg_to_user
