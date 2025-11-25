from pydantic import BaseModel

from .llm import structured_agent_response
from .types import History, Message, MessageFlag, OpenAiClientConfig


class ConvSummary(BaseModel):
    summary: str


def last_summary_index(history: History) -> int:
    return next(
        (
            idx
            for idx, message in reversed(list(enumerate(history.root)))
            if MessageFlag.is_summary in message.flags
        ),
        0,
    )


async def create_summary_entry(
    old_history: History,
    reduce_by: int,
    client_config: OpenAiClientConfig | None = None,
) -> History:
    """Create a summary entry for the conversation history."""
    history = old_history.model_copy(deep=True)

    truncated_conv = "\n".join(
        f"{msg.role}: {msg.content}" for msg in history.root[:reduce_by]
    )

    prompt = f"Please summarize the following conversation between the user and the assistant in a concise manner, retaining all important details:\n\n{truncated_conv}"

    truncation_hist = History.model_validate([Message(role="user", content=prompt)])
    summary_response = await structured_agent_response(
        history=truncation_hist,
        schema=ConvSummary,
        client_config=client_config,
    )
    history.add_message(
        role="system",
        content=f"Summary of previous conversation: {summary_response.summary}",
        flags=[MessageFlag.is_summary],
        index=reduce_by + last_summary_index(history),
    )
    return history
