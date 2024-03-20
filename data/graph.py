import dataclasses


from typing import Union, Optional, List
from pydantic import BaseModel

from data.chat import Role


@dataclasses.dataclass
class MessageOutput:
    message: str
    role: Role


@dataclasses.dataclass
class EdgeOutput:
    should_continue: bool
    result: Union[BaseModel, str]
    message_output: Optional[List[MessageOutput]]
    num_fails: int
    next_node: "BaseNode"
