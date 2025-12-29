from pathlib import Path

import pytest
from pydantic import BaseModel

from wingmate.llm import stream_agent_response
from wingmate.types import History, OpenAiClientConfig


class SchemaForTest(BaseModel):
    thought: str
    response: str


@pytest.mark.asyncio
async def test_simulated_stream():
    history = History(root=())
    data_path = Path(__file__).parent / "data" / "stream_test.txt"

    client_config = OpenAiClientConfig(
        llm_model_name="newline",
        base_url=f"file:{data_path}",
        api_key="dummy",
    )

    results: list[SchemaForTest] = []
    async for chunk in stream_agent_response(history, SchemaForTest, client_config):
        results.append(chunk)

    assert len(results) > 0
    final_result = results[-1]
    assert final_result.thought == "I am thinking"
    assert final_result.response == "Here is the answer"
