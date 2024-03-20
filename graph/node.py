import abc
from typing import List, Generic, TypeVar, Union, Optional

from data.graph import EdgeOutput, MessageOutput
from graph.edge import BaseEdge


NodeInput = TypeVar("NodeInput")


class BaseNode(abc.ABC, Generic[NodeInput]):

    """Node
    at it's highest level, a node asks a user for some input, and trys
    that input on all edges. It also manages and executes all
    the edges it contains
    """

    def __init__(self, edges: Optional[List[BaseEdge]] = None, final_state=False):
        """
        prompt (str): what to ask the user
        retry_prompt (str): what to ask the user if all edges fail
        parse_class (Pydantic BaseModel): the structure of the parse
        llm (LangChain LLM): the large language model being used
        """

        self._edges = edges
        self._node_input = None
        self._final_state = final_state

    def is_node_final(self):
        return self._final_state

    def set_node_input(self, edge_output: EdgeOutput):
        self._node_input = edge_output

    def run_to_continue(self, user_input: NodeInput) -> Optional[EdgeOutput]:
        """Run all edges until one continues
        returns the result of the continuing edge, or None
        """
        res = None
        for edge in self._edges:
            res = edge.execute(user_input)
            if res is not None and res.should_continue:
                return res
        return res

    def execute(self, user_input: NodeInput) -> Union[MessageOutput, EdgeOutput]:
        """Handles the current conversational state
        prompts the user, tries again, runs edges, etc.
        returns the result from an adge
        """
        res = self.run_to_continue(user_input)
        if res is None or not res.should_continue:
            return self.no_edges_found(user_input)
        else:
            if res.next_node is not None:
                res.next_node.set_node_input(res.result)

        return res

    @abc.abstractmethod
    def greeting_message(self) -> Optional[MessageOutput]:
        pass

    @abc.abstractmethod
    def no_edges_found(self, user_input: NodeInput) -> Optional[MessageOutput]:
        pass
