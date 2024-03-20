import abc

from typing import List, Optional


from data.graph import EdgeOutput
from graph.edge import BaseEdge
from graph.node import BaseNode


class StaticTextNode(BaseNode[str]):
    def __init__(self, edges: Optional[List[BaseEdge]] = None):
        super().__init__(edges=edges)

    @abc.abstractmethod
    def _node_static_prompt(self, **kwargs) -> str:
        pass

    @abc.abstractmethod
    def _node_static_retry(self, **kwargs) -> str:
        pass

    def greeting_message(self):
        return self._node_static_prompt()

    def no_edges_found(self, user_input: str) -> EdgeOutput:
        return EdgeOutput(
            should_continue=False,
            result=self._node_static_retry(),
            num_fails=0,
            next_node=None,
        )
