import logging

from pydantic import BaseModel

from .environment import Environment
from .llm import stream_agent_response
from .settings import settings
from .types import (
    TERMINATE,
    MaxAgentIterationsExceededError,
    OpenAiClientConfig,
)
from .utils import build_agent_response_schema

logger = logging.getLogger(__name__)


class LocalAgent[T: BaseModel]:
    """
    An agent that orchestrates LLM interactions with tool calling capabilities.
    """

    def __init__(
        self,
        environment: Environment[type[T]],
        disable_thought: bool = True,
        max_iterations: int = settings.max_agent_iterations,
        message_separation_token: str = "\n\n",
        openai_client: OpenAiClientConfig | None = None,
    ):
        """
        Initialize the LocalAgent.

        Args:
            environment: Environment instance that defines tools, context engineering,
                tool execution, and termination logic.
            max_iterations: Maximum number of agent loop iterations to prevent infinite loops.
                Defaults to settings.max_agent_iterations.
            message_separation_token: Token used to separate messages when streaming responses.
                Defaults to "\n\n".
            openai_client: Optional AsyncOpenAI client for LLM interactions. If not provided, a default client will be created using the config provided in `local-agent-config.yaml`.
        """
        self.environment = environment
        self.disable_thought = disable_thought
        self.max_iterations = max_iterations
        self.message_separation_token = message_separation_token
        self.openai_client = openai_client

    async def run(self):
        """
        Run the agent loop for a given user query.

        This method implements the agent's main loop:
        - Applies context engineering via environment
        - Streams LLM responses
        - Handles tool calls via environment
        - Continues loop until environment signals termination

        Args:
            history: The conversation history to process

        Yields:
            AgentResponse objects as they are generated during the agent's execution

        Raises:
            MaxAgentIterationsExceededError: If the agent exceeds max_iterations
        """

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1

            # Context Engineering via Environment
            context = await self.environment.get_context(
                self.max_iterations - iteration
            )

            response = None
            schema = build_agent_response_schema(
                disable_thought=self.disable_thought,
                tools=await self.environment.get_tools(),
            )
            async for response in stream_agent_response(
                context, schema, self.openai_client
            ):
                yield response

            # Agent Message Completion Hook via Environment
            assert response, "Agent failed to produce a response"
            self.environment.history.add_message(
                role="assistant",
                content=response.model_dump_json(
                    indent=2,
                    exclude_defaults=True,
                    exclude={"flags"},
                    exclude_none=True,
                    exclude_unset=True,
                ),
            )

            # Environment handles tool execution and continuation decision
            response.turn_completed = True
            yield response
            continuation_message = await self.environment.on_agent_message_completed(
                response
            )

            if continuation_message is TERMINATE:
                return

            # Add continuation message to history
            self.environment.history.add_message(continuation_message)

        raise MaxAgentIterationsExceededError(
            f"Agent exceeded maximum iterations ({self.max_iterations})"
        )

    async def stream_text(self):
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
        async for response in self.run():
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
