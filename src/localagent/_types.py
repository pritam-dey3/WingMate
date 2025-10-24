from typing import Literal

from pydantic import BaseModel, RootModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


History = RootModel[list[Message]]
