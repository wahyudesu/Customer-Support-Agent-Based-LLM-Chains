import abc
from abc import ABC
from typing import Type, Optional, Union

from langchain.agents import ZeroShotAgent, AgentExecutor, AgentType, initialize_agent
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BasePromptTemplate
from pydantic import BaseModel

from data.chat import MessageHistory, ModelInput
from data.graph import MessageOutput
from graph.edge import BaseEdge


class ChainBasedEdge(BaseEdge[MessageHistory, MessageOutput], ABC):
    def __init__(
        self,
        model,
        pydantic_object: Optional[Type[BaseModel]],
        max_retries=3,
        out_node=None,
    ):
        super().__init__(model=model, max_retries=max_retries, out_node=out_node)
        if pydantic_object is not None:
            self._output_parser = PydanticOutputParser(pydantic_object=pydantic_object)
        else:
            self._output_parser = None

        self._init_chain()

    @abc.abstractmethod
    def _predict(self, model_input: ModelInput) -> str:
        pass

    @abc.abstractmethod
    def _init_chain(self, *kwargs):
        pass

    @abc.abstractmethod
    def _get_prompt_template(self) -> BasePromptTemplate:
        pass

    @abc.abstractmethod
    def _prompt_input_variables(self) -> list:
        pass

    def check(self, model_output: str) -> bool:
        return isinstance(self._output_parser.parse(model_output), BaseModel)

    def _parse(self, message_history: MessageHistory) -> Union[str, BaseModel]:
        model_input = message_history.model_input()
        str_to_parse = self._predict(model_input=model_input)
        out = (
            self._output_parser.parse(str_to_parse)
            if self._output_parser is not None
            else str_to_parse
        )
        return out


class ZeroShotChainBasedEdge(ChainBasedEdge, ABC):
    _prompt_prefix = None
    _prompt_suffix = None

    def _prompt_input_variables(self):
        input_variables = ["input", "agent_scratchpad", "history"]
        if self._output_parser is not None:
            input_variables += ["format_instructions"]

        return input_variables

    def _get_prompt_template(self) -> BasePromptTemplate:
        input_variables = self._prompt_input_variables()
        history = """Conversation History \n{history}\n"""

        prompt = ZeroShotAgent.create_prompt(
            tools=self._tools,
            prefix=self._prompt_prefix,
            suffix=history + self._prompt_suffix,
            input_variables=input_variables,
        )
        return prompt

    def _init_chain(self, **kwargs):
        self._tools = self._get_tools()

        self._prompt = self._get_prompt_template()
        self._llm_chain = LLMChain(llm=self._llm_model, prompt=self._prompt)

        agent = ZeroShotAgent(llm_chain=self._llm_chain, tools=self._tools)
        self._agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=self._tools, verbose=True, handle_parsing_errors=True
        )

    @abc.abstractmethod
    def _get_tools(self):
        pass

    def _predict(self, model_input: ModelInput) -> str:
        if self._output_parser is not None:
            result = self._agent_executor.run(
                input=model_input.input,
                history=model_input.history,
                format_instructions=self._output_parser.get_format_instructions(),
            )
        else:
            result = self._agent_executor.run(
                input=model_input.input, history=model_input.history
            )

        return result


class MultifunctionEdge(ChainBasedEdge, ABC):
    _prompt_prefix = None
    _prompt_suffix = None

    def _init_chain(self, *kwargs):
        self._tools = self._get_tools()

        self._agent = initialize_agent(
            self._tools,
            ChatOpenAI(temperature=0),
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
        )

    @abc.abstractmethod
    def _get_tools(self):
        pass

    def _predict(self, messages: MessageHistory) -> str:
        completion = self._agent.run(messages)
        return completion
