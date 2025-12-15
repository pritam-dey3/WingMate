import asyncio

from fastmcp import Client, FastMCP

from localagent import DefaultEnvironment, LocalAgent
from localagent.types import BaseToolModel, CallToolRequestParams
from localagent.utils import mcp_tools

## Define the tools for date calculations
## LocalAgent supports any MCP servers
mcp = FastMCP()


@mcp.tool
def day_of_date(date: str) -> str:
    """Get the day of the week for a given date string in YYYY-MM-DD format."""
    import datetime

    dt = datetime.datetime.strptime(date, "%Y-%m-%d")
    return dt.strftime("%A")


@mcp.tool
def days_between_dates(start_date: str, end_date: str) -> int:
    """Calculate the number of days between two dates in YYYY-MM-DD format."""
    import datetime

    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    delta = end_dt - start_dt
    return abs(delta.days)


## Create the MCP client
client = Client(mcp)


## Define environment and agent
class SimpleEnvironment[T: BaseToolModel](DefaultEnvironment[T]):
    async def call_tool(self, action: CallToolRequestParams[T]) -> str | None:
        async with client:
            result = await client.call_tool(
                name=action.tool_name, arguments=action.arguments.model_dump()
            )
        return "\n".join(res.text for res in result.content if res.type == "text")


env = SimpleEnvironment(tools=mcp_tools(client))
agent = LocalAgent(
    environment=env, disable_thought=True, require_terminating_tool_call=False
)


async def main():
    user_query = input("Query: ")
    # Add user query to history
    agent.environment.history.add_message(role="user", content=user_query)
    async for event in agent.stream():
        print(event.model_dump_json(indent=2))
        print("------------")


asyncio.run(main())
