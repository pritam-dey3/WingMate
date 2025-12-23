# WingMate

**Orchestrate AI agents locally with ease.**

WingMate is a lightweight, flexible Python framework for building and orchestrating AI agents. It is designed to run locally, with a tool definition system closely aligned with the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), making it very simple to integrate MCP servers.
## Features

- **Local-First**: Designed to run agents in your local environment.
- **MCP Integration**: Seamlessly integrate with MCP servers and tools.
- **Thoughtful Agents**: Optional "thought" process visibility for debugging and transparency.
- **Highly Configurable**: Easy configuration via YAML or environment variables.
- **Streaming**: Built-in support for streaming agent responses.
- **Extensible**: Create custom environments to control tool execution and context.

## Installation

You can install Wingmate directly from GitHub:

```bash
pip install wingmate
```

## Quick Start

Here is a simple example of how to create an agent that can perform date calculations using MCP tools.

### 1. Define Tools & Environment

Create a file named `main.py`:

```python
import asyncio
from fastmcp import Client, FastMCP
from wingmate import DefaultEnvironment, Agent
from wingmate.types import BaseToolModel, CallToolRequestParams
from wingmate.utils import mcp_tools

# 1. Define tools using FastMCP
mcp = FastMCP()

@mcp.tool
def day_of_date(date: str) -> str:
    """Get the day of the week for a given date string in YYYY-MM-DD format."""
    import datetime
    dt = datetime.datetime.strptime(date, "%Y-%m-%d")
    return dt.strftime("%A")

# 2. Create an MCP client
client = Client(mcp)

# 3. Define the Environment
# The environment handles tool execution and context management
class SimpleEnvironment[T: BaseToolModel](DefaultEnvironment[T]):
    async def call_tool(self, action: CallToolRequestParams[T]) -> str | None:
        async with client:
            result = await client.call_tool(
                name=action.tool_name,
                arguments=action.arguments.model_dump()
            )
        return "\n".join(res.text for res in result.content if res.type == "text")

# 4. Initialize Agent
async def main():
    # Initialize environment with tools from the MCP client
    env = SimpleEnvironment(tools=mcp_tools(client))

    # Create the agent
    agent = Agent(
        environment=env,
        disable_thought=False  # Set to False to see the agent's thinking process
    )

    # Add a user message to the history
    env.history.add_message(role="user", content="What day was it on January 1st, 2000?")

    # Run the agent
    async for event in agent.stream():
        print(event.model_dump_json(indent=2))
        print("------------")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Configure LLM

Create a `wingmate-config.yaml` file in your project root to configure your LLM provider (e.g., OpenAI or OpenAI-compatible):

```yaml
llm_model_name: "gpt-4o"
llm_api_key: "your-api-key-here"
# Optional: Base URL for other compatible providers
# llm_base_url: "https://api.openai.com/v1"
```

### 3. Run

```bash
python main.py
```

## Configuration

Wingmate uses `pydantic-settings` for configuration. You can configure it using a `wingmate-config.yaml` file, a `.env` file, or environment variables.

| Setting                | Description                                    | Default |
| ---------------------- | ---------------------------------------------- | ------- |
| `llm_model_name`       | The name of the LLM model to use.              | `None`  |
| `llm_api_key`          | API key for the LLM provider.                  | `None`  |
| `llm_base_url`         | Base URL for the LLM API.                      | `None`  |
| `max_agent_iterations` | Maximum number of loops the agent can perform. | `7`     |
| `max_history_length`   | Maximum number of messages to keep in history. | `11`    |
| `llm_api_extra_kw`     | Extra keyword arguments for the LLM API call.  | `{}`    |

## Advanced Usage

### Custom Environments

The `Environment` class is the heart of Wingmate's extensibility. By subclassing `DefaultEnvironment` or implementing the `Environment` protocol, you can:

- **Customize Tool Execution**: Handle tool calls locally, remotely, or via complex pipelines.
- **Manage Context**: Control how history is stored, retrieved, and presented to the LLM.
- **Implement Termination Logic**: Define custom criteria for when the agent should stop.

### Thought Process

Wingmate can expose the agent's internal "thought" process. When `disable_thought=False` is passed to the `Wingmate` constructor, the agent will generate a thought trace before taking actions or answering. This is useful for debugging and understanding the agent's reasoning.

## License

[MIT](LICENSE)
