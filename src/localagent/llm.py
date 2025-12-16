import logging

from openai import AsyncOpenAI
from partialjson.json_parser import JSONParser
from pydantic import BaseModel

from .settings import settings
from .types import History, OpenAiClientConfig

logger = logging.getLogger(__name__)

parser = JSONParser()


async def stream_agent_response[T: BaseModel](
    history: History,
    schema: type[T],
    client_config: OpenAiClientConfig | None = None,
):
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

    content = ""
    last_yielded = schema()

    async with client.responses.stream(
        model=client_config.llm_model_name,
        input=history.compact(),  # type: ignore
        text_format=schema,
        extra_body=client_config.extra_kw,
    ) as stream:
        async for event in stream:
            if (
                event.type == "response.output_text.delta"
                or event.type == "response.refusal.delta"
            ):
                content += event.delta
                try:
                    parsed = parser.parse(content)
                    result = schema.model_validate(parsed)
                    if result != last_yielded:
                        last_yielded = result
                        yield result
                except Exception:
                    continue


async def structured_agent_response[T](
    history: History,
    schema: type[T],
    client_config: OpenAiClientConfig | None = None,
) -> T:
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

    response = await client.responses.parse(
        model=client_config.llm_model_name,
        input=history.model_dump(),
        text_format=schema,
        extra_body=client_config.extra_kw,
    )

    assert response.output_parsed is not None, "Expected parsed response to be present"
    return response.output_parsed
