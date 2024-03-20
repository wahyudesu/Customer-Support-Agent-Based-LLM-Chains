import abc
from typing import Type, Optional, Union

from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel

from data.chat import MessageHistory, Role
from data.graph import MessageOutput
from data.validation import Validation
from graph.edge import BaseEdge


class PydanticTextBasedEdge(BaseEdge[MessageHistory, MessageOutput]):

    """Edge
    at it's highest level, an edge checks if an input is good, then parses
    data out of that input if it is good
    """

    def __init__(
        self,
        condition: str,
        parse_prompt: str,
        parse_class: Type[BaseModel],
        llm_model,
        max_retries: Optional[int] = None,
        out_node=None,
    ):
        """
        condition (str): a True/False question about the input
        parse_query (str): what the parser whould be extracting
        parse_class (Pydantic BaseModel): the structure of the parse
        llm (LangChain LLM): the large language model being used
        """
        super().__init__(model=llm_model, max_retries=max_retries, out_node=out_node)
        self.condition = condition
        self.parse_prompt = parse_prompt
        self.parse_class = parse_class
        self._validation_parser = PydanticOutputParser(pydantic_object=Validation)
        self._extraction_parser = PydanticOutputParser(pydantic_object=self.parse_class)
        self._validation_llm_chain = LLMChain(
            llm=llm_model, prompt=self._get_validation_prompt_template()
        )
        self._extraction_llm_chain = LLMChain(
            llm=llm_model, prompt=self._get_extraction_prompt_template()
        )

    def _get_validation_prompt_template(self):
        model_input = (
            "Answer the user query."
            "\n{format_instructions}"
            "\nConversation history:"
            "\n{history}"
            "\nFollowing the output schema, does the input satisfy the condition?"
            "\nCondition: {condition}"
            "\nInput: {query}"
        )

        prompt = PromptTemplate(
            template=model_input,
            input_variables=["condition", "history"],
            partial_variables={
                "format_instructions": self._validation_parser.get_format_instructions()
            },
        )
        return prompt

    def _get_extraction_prompt_template(self):
        parse_query = "{parse_prompt}:" "\n{format_instructions}" "\n\nInput: {query}"

        prompt = PromptTemplate(
            template=parse_query,
            input_variables=["parse_prompt"],
            partial_variables={
                "format_instructions": self._extraction_parser.get_format_instructions()
            },
        )
        return prompt

    def check(self, user_input: MessageHistory) -> bool:
        """ask the llm if the input satisfies the condition"""
        history = "\n".join((str(user_input)).split("\n")[:-1])
        last_input = (user_input.role_based_history(Role.USER)[-1])["content"]

        completion = self._validation_llm_chain.run(
            query=last_input, condition=self.condition, history=history
        )
        return self._validation_parser.parse(completion).is_valid

    def _parse(self, user_input: MessageHistory) -> Union[str, BaseModel]:
        """ask the llm to parse the parse_class, based on the parse_prompt, from the input"""
        completion = self._extraction_llm_chain.run(
            query=user_input, parse_prompt=self.parse_prompt
        )
        base_model = self._extraction_parser.parse(completion)

        return base_model

    def _predict(self, model_input: MessageHistory) -> str:
        return self._llm_model(model_input)

    def execute(self, user_input: MessageHistory):
        # input did't make it past the input condition for the edge
        if not self.check(user_input):
            self._num_fails += 1
            if self._max_retries is not None:
                if self._num_fails >= self._max_retries:
                    return self._get_edge_output(should_continue=True, result=None)
            return self._get_edge_output(should_continue=False, result=None)
        return super().execute(user_input)
