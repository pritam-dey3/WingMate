import logging
from pathlib import Path

from openai import AsyncOpenAI
from partialjson.json_parser import JSONParser
from pydantic import BaseModel

from .settings import settings
from .types import History, OpenAiClientConfig

logger = logging.getLogger(__name__)

parser = JSONParser()


class StructuredStreamParser[T: BaseModel]:
    def __init__(self, schema: type[T]):
        self.schema = schema
        self.buffer = ""

    def feed(self, chunk: str) -> T | None:
        self.buffer += chunk
        try:
            parsed = parser.parse(self.buffer)
            result = self.schema.model_validate(parsed)
            return result
        except Exception as e:
            logger.debug(f"Stream parsing error: {e}\nBuffer: {self.buffer}")
            return None


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
    stream_parser = StructuredStreamParser(schema)
    last_yielded: T | None = None

    if client_config.base_url.startswith("file:"):
        logger.debug(
            f"Using simulated agent stream from file. {client_config.base_url[5:]}"
        )
        async for chunk in simulated_agent_stream(
            Path(client_config.base_url[5:]),
            newline_delimited=client_config.llm_model_name == "newline",
        ):
            parsed = stream_parser.feed(chunk)
            if parsed is not None and parsed != last_yielded:
                last_yielded = parsed
                yield parsed
        return
    client = AsyncOpenAI(
        base_url=client_config.base_url,
        api_key=client_config.api_key,
    )

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
                parsed = stream_parser.feed(event.delta)
                if parsed is not None and parsed != last_yielded:
                    last_yielded = parsed
                    yield parsed
                # content += event.delta
                # try:
                #     parsed = parser.parse(content)
                #     result = schema.model_validate(parsed)
                #     if result != last_yielded:
                #         last_yielded = result
                #         yield result
                # except Exception:
                #     continue


async def simulated_agent_stream(path: Path, newline_delimited: bool = True):
    """Simulates streaming by reading a file and yielding its content in chunks."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if newline_delimited:
        for line in content.splitlines(keepends=True):
            yield line
    else:
        import regex

        _unused_pat = regex.compile(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
        )
        for piece in regex.findall(_unused_pat, content):
            yield piece


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
