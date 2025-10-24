from typing import Callable, Coroutine, Sequence, Union

from ._types import Action, ActionResult, AgentResponse, History, Message, Token
from .llm import respond


class FinishTurn(Action):
    action_name = "finish_turn"
    user_query_resolved: bool


async def get_agent_response(
    history: History,
    actions: Sequence[Action],
    action_callback: Callable[[Action], Coroutine[None, None, ActionResult]],
):
    available_actions = Union[tuple(actions) + (FinishTurn,)]
    while True:
        last_response = None
        async for response in respond(history, AgentResponse[available_actions]):  # type: ignore
            if last_response and last_response.msg_to_user:
                assert response.msg_to_user and len(response.msg_to_user) >= len(
                    last_response.msg_to_user
                ), (
                    "msg_to_user should not be None if last_response.msg_to_user is not None"
                )
                delta = response.msg_to_user[len(last_response.msg_to_user) :]
                if delta:
                    yield Token(token=delta)
            else:
                if response.msg_to_user:
                    yield Token(token=response.msg_to_user)

            last_response = response

        if not last_response:
            raise RuntimeError("No response from agent")

        action_impact = ""
        if isinstance(last_response.action, FinishTurn):
            break
        elif last_response.action:
            try:
                action_call = await action_callback(last_response.action)  # type: ignore
                action_impact = f"\nAction: {action_call.action_name}\n Result: {action_call.result}"
            except Exception as e:
                action_impact = f"\nAction: {last_response.action.action_name}\n Result: Failed to execute action: {e}"  # type: ignore
            yield last_response.action

        msg = Message(
            role="assistant",
            content=f"Thought: {last_response.thought}\nMsg to user: {last_response.msg_to_user}{action_impact}",
        )
        history.root.append(msg)
