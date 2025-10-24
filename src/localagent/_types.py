from typing import Literal

from pydantic import BaseModel, RootModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


History = RootModel[list[Message]]


class Action(BaseModel):
    action_name: str


class ActionResult(BaseModel):
    action_name: str
    result: str


class AgentResponse[A: Action](BaseModel):
    """Response for the given user query, including the agent's thought process"""

    thought: str
    msg_to_user: str | None
    action: A | None


class Token(BaseModel):
    token: str
