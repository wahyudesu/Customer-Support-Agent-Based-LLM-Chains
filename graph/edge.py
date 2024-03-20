import abc
from typing import Generic, TypeVar, Optional, Union, List

from langchain.schema import OutputParserException
from pydantic import BaseModel

from data.chat import Role
from data.graph import EdgeOutput, MessageOutput

EdgeInput = TypeVar("EdgeInput")
ResultsType = TypeVar("ResultsType")


class BaseEdge(abc.ABC, Generic[EdgeInput, ResultsType]):
    def __init__(self, model, max_retries=3, out_node=None):
        self._llm_model = model

        # how many times the edge has failed, for any reason, for deciding to skip
        # when successful this resets to 0 for posterity.
        self._num_fails = 0

        # how many retrys are acceptable
        self._max_retries = max_retries

        # the node the edge directs towards
        self._out_node = out_node

    @abc.abstractmethod
    def _get_message_output(
        self, msg_input: Union[str, BaseModel]
    ) -> Optional[List[MessageOutput]]:
        pass

    @abc.abstractmethod
    def check(self, model_output: str) -> bool:
        pass

    @abc.abstractmethod
    def _parse(self, model_input: EdgeInput) -> ResultsType:
        pass

    def _get_edge_output(
        self, should_continue: bool, result: Optional[ResultsType]
    ) -> EdgeOutput:
        message_output = self._get_message_output(result)
        return EdgeOutput(
            should_continue=should_continue,
            result=result,
            num_fails=self._num_fails,
            next_node=self._out_node,
            message_output=message_output,
        )

    def execute(self, user_input: EdgeInput):
        """Executes the entire edge
        returns a dictionary:
        {
            continue: bool,       weather or not should continue to next
            result: parse_class,  the parsed result, if applicable
            num_fails: int        the number of failed attempts
            continue_to: Node     the Node the edge continues to
        }
        """

        try:
            # attempting to parse
            self._num_fails = 0
            return self._get_edge_output(
                should_continue=True, result=self._parse(user_input)
            )
        except OutputParserException as parsing_exception:
            # there was some error in parsing.
            # note, using the retry or correction parser here might be a good idea
            self._num_fails += 1
            if self._num_fails >= self._max_retries:
                return self._get_edge_output(
                    should_continue=True,
                    result=MessageOutput(
                        parsing_exception.llm_output, role=Role.SYSTEM
                    ),
                )
            return self._get_edge_output(
                should_continue=False,
                result=MessageOutput(parsing_exception.llm_output, role=Role.SYSTEM),
            )
