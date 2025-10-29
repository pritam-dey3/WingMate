from typing import AsyncGenerator, Type

from openai import AsyncOpenAI
from partialjson.json_parser import JSONParser
from pydantic import BaseModel

from ._types import History
from .settings import settings

parser = JSONParser(strict=False)
client = AsyncOpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
)


async def respond[T: BaseModel](
    history: History, schema: Type[T]
) -> AsyncGenerator[T, None]:
    response = await client.chat.completions.create(
        model=settings.llm_model_name,
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
        extra_body=settings.llm_api_extra_kw,
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
