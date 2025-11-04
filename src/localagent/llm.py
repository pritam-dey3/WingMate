from typing import AsyncGenerator, Type

from openai import AsyncOpenAI
from partialjson.json_parser import JSONParser
from pydantic import BaseModel

from .settings import settings
from .types import History, OpenAiClientConfig

parser = JSONParser(strict=False)


async def stream_agent_response[T: BaseModel](
    history: History,
    schema: Type[T],
    client_config: OpenAiClientConfig | None = None,
) -> AsyncGenerator[T, None]:
    if client_config is None:
        assert settings.llm_model_name is not None, (
            "llm_model_name must be set in `local-agent-config.yaml`"
        )
        assert settings.llm_base_url is not None, (
            "llm_base_url must be set in `local-agent-config.yaml`"
        )
        client_config = OpenAiClientConfig(
            llm_model_name=settings.llm_model_name,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            extra_kw=settings.llm_api_extra_kw,
        )
    client = AsyncOpenAI(
        base_url=client_config.base_url,
        api_key=client_config.api_key,
    )
    response = await client.chat.completions.create(
        model=client_config.llm_model_name,
        messages=history.model_dump(),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema.__class__.__name__,
                "description": schema.__doc__ or "",
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        },
        stream=True,
        extra_body=client_config.extra_kw,
    )

    content = ""
    last_yielded = schema()
    async for chunk in response:
        if not chunk.choices[0].delta.content:
            continue
        content += chunk.choices[0].delta.content
        try:
            parsed = parser.parse(content)
            result = schema.model_validate(parsed)
            if result != last_yielded:
                last_yielded = result
                yield result
        except Exception:
            continue
