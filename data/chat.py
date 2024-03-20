import dataclasses
import json

from enum import Enum
from typing import List


class Role(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"

    def __str__(self):
        return str(self.value)


@dataclasses.dataclass
class ModelInput:
    input: str
    history: str


@dataclasses.dataclass
class MessageHistory:
    messages: List[dict]

    def __str__(self):
        history = ""
        for msg in self.messages:
            history += f"\n{msg['role']}: {msg['content']}"
        return history

    def model_input(self) -> ModelInput:
        history = ""
        for msg in self.messages[:-1]:
            history += f"\n{msg['role']}: {msg['content']}"

        user_messages = self.role_based_history(role=Role.USER)
        last_msg = user_messages[-1]
        user_input = f"\n{last_msg['role']}: {last_msg['content']}"

        return ModelInput(input=user_input, history=history)

    def role_based_history(self, role: Role):
        history = []
        for msg in self.messages:
            if msg["role"] == role:
                history.append(msg)
        return history

    @classmethod
    def _message_dict(self, content: str, role: Role):
        return {"content": content, "role": role.value}

    def add_system_message(self, content: str):
        self.messages.append(self._message_dict(content=content, role=Role.SYSTEM))

    def add_user_message(self, content: str):
        self.messages.append(self._message_dict(content=content, role=Role.USER))

    def add_assistant_message(self, content: str):
        self.messages.append(self._message_dict(content=content, role=Role.ASSISTANT))

    def add_message(self, content: str, role: Role):
        self.messages.append(self._message_dict(content=content, role=role))
