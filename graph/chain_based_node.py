import abc

from langchain.agents import initialize_agent, AgentType
from langchain.chains import MultiRetrievalQAChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import Type, Optional, List

from data.chat import MessageHistory
from graph.node import BaseNode
from graph.edge import BaseEdge


class ChainBasedNode(BaseNode[MessageHistory], abc.ABC):
    def __init__(
        self,
        llm_model,
        pydantic_object: Optional[Type[BaseModel]],
        edges: Optional[List[BaseEdge]],
        final_state=False,
    ):
        self._llm_model = llm_model
        self._parse_class = pydantic_object

        if pydantic_object is not None:
            self._output_parser = PydanticOutputParser(pydantic_object=pydantic_object)
        else:
            self._output_parser = None

        self._init_chain()
        super().__init__(edges, final_state)

    @abc.abstractmethod
    def _init_chain(self, **kwargs):
        pass


class MultiRetrievalNode(ChainBasedNode, abc.ABC):
    @abc.abstractmethod
    def _get_retriever_infos(self):
        pass

    @abc.abstractmethod
    def _get_default_chain(self):
        pass

    def _init_chain(self, *kwargs):
        retriever_infos = self._get_retriever_infos()

        self._llm_chain = MultiRetrievalQAChain.from_retrievers(
            self._llm_model,
            retriever_infos,
            default_chain=self._get_default_chain(),
            verbose=True,
        )

    def _predict(self, messages: MessageHistory) -> str:
        return self._llm_chain.run(messages)


class MultifunctionNode(ChainBasedNode, abc.ABC):
    def _init_chain(self, *kwargs):
        self._tools = self._get_tools()

        self._agent = initialize_agent(
            self._tools, self._llm_model, agent=AgentType.OPENAI_FUNCTIONS, verbose=True
        )

    @abc.abstractmethod
    def _get_tools(self):
        pass

    def _predict(self, messages: MessageHistory) -> str:
        completion = self._agent.run(messages)
        return completion
