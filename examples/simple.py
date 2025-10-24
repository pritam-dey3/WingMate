from typing import Union

from localagent._types import Action, AgentResponse


class Answer(Action):
    query_resolved: bool


class Question(Action):
    pass


actions = [Answer, Question]
MyReq = AgentResponse[Union[tuple(actions)]]
print(MyReq.model_json_schema())
