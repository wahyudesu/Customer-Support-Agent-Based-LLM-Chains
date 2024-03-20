import random
from typing import Optional, Type, Union, List

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from pydantic import BaseModel, Field

from data.chat import MessageHistory, Role
from data.graph import MessageOutput
from data.validation import UserProfile, PhoneCallRequest, PhoneCallTicket
from graph.chain_based_edge import ZeroShotChainBasedEdge
from graph.chain_based_node import MultiRetrievalNode, MultifunctionNode
from graph.node import BaseNode, BaseEdge, NodeInput
from graph.static_text_node import StaticTextNode
from graph.text_based_edge import PydanticTextBasedEdge
from tools.audio_transcribe import call_customer
from tools.rag_responder import HelpCenterAgent
from tools.user_info_db import search_user_info_on_db, search_user_subscription_on_db


class GreetingNode(BaseNode[str]):
    STATIC_PROMPT = [
        "Hi, welcome to our online support, in order to proceed we need to identify you first, "
        "could you please input your full email address or phone number"
    ]
    RETRY_PROMPT = [
        "I'm sorry, I didn't understand your response."
        "\nPlease provide a full email address or phone number(in the format xxx-xxx-xxxx)"
    ]

    def greeting_message(self) -> Optional[MessageOutput]:
        prompt = random.choice(self.STATIC_PROMPT)
        return MessageOutput(prompt, role=Role.ASSISTANT)

    def no_edges_found(self, **kwargs) -> Optional[MessageOutput]:
        prompt = random.choice(self.RETRY_PROMPT)
        return MessageOutput(prompt, role=Role.ASSISTANT)


class UserInfoChainBasedEdge(ZeroShotChainBasedEdge):
    _prompt_prefix = """Your goal is to find out the user information and their subscription type.
- You MUST ALWAYS combine the output of different tools to achieve your final answer.
- The user subscription must be either free or premium, never empty

To achieve this you have access to the following tools:"""

    _prompt_suffix = """\nYour final answer should combine the information of previous Observations 
{format_instructions}
Begin! 
Question: {input}
{agent_scratchpad}
"""

    def _get_tools(self):
        tools = [
            Tool.from_function(
                func=search_user_info_on_db,
                description="Database tool to search user information, input should be their email as text",
                name="user_info_db_search",
            ),
            Tool.from_function(
                func=search_user_subscription_on_db,
                description="Database tool to search user subscription type by user id, requires a number as input,",
                name="user_subscription_db_search",
            ),
        ]
        return tools

    def _get_message_output(
        self, msg_input: Union[str, BaseModel]
    ) -> List[MessageOutput]:
        user_info = msg_input if isinstance(msg_input, str) else str(msg_input)
        message = f"User Info retrieved: {user_info}"
        return [MessageOutput(message, Role.SYSTEM)]


class AuthenticatedUserNode(MultiRetrievalNode):
    STATIC_PROMPT = [
        "Hi, {user_name} I am your Shopify Agent for today, you have the {subscription} subscription "
        "I can help you with any Help or you can ask me to call you at anytime!"
    ]

    def __init__(
        self,
        llm_model,
        pydantic_object: Optional[Type[BaseNode]],
        edges: List[BaseEdge] = None,
    ):
        self._hc_agent = HelpCenterAgent()
        super().__init__(llm_model, pydantic_object, edges)

    def greeting_message(self) -> Optional[MessageOutput]:
        prompt = random.choice(self.STATIC_PROMPT)
        user_profile: UserProfile = self._node_input

        prompt = prompt.format(
            user_name=user_profile.name, subscription=user_profile.subscription
        )
        return MessageOutput(prompt, role=Role.ASSISTANT)

    def _get_retriever_infos(self):
        retriever_infos = [
            {
                "name": "Premium Subscription Knowledge Base",
                "description": "Contains information for user with a premium subscription",
                "retriever": self._hc_agent.paid_sub_retriever(),
            },
            {
                "name": "Free Subscription Knowledge Base",
                "description": "Contains information for user with a free subscription",
                "retriever": self._hc_agent.free_sub_retriever(),
            },
        ]
        return retriever_infos

    def _get_default_chain(self):
        template = """You are a helpful assistant, you should tell the user that his query is outside of your domain 
    in a friendly way"
    Human: """

        prompt_template = PromptTemplate.from_template(template)
        chain = LLMChain(
            llm=self._llm_model, prompt=prompt_template, output_key="result"
        )
        return chain

    def no_edges_found(self, user_input: MessageHistory) -> Optional[MessageOutput]:
        message = self._predict(user_input)
        return MessageOutput(message=message, role=Role.ASSISTANT)


class CallCustomerEdge(PydanticTextBasedEdge):
    def __init__(self, llm_model, max_retries: int = 3, out_node: BaseNode = None):
        super().__init__(
            condition="Is there any pending call requests coming from the user?",
            parse_prompt="Extract the phone number from the user message",
            parse_class=PhoneCallRequest,
            llm_model=llm_model,
            max_retries=max_retries,
            out_node=out_node,
        )

    def _get_message_output(
        self, msg_input: Union[str, BaseModel]
    ) -> Optional[List[MessageOutput]]:
        if isinstance(msg_input, PhoneCallRequest):
            system_message = MessageOutput(
                f"User has been called as per their request", Role.SYSTEM
            )
            assistant_message = MessageOutput(
                f"Sure we are calling you now on: {msg_input.phone_number}",
                Role.ASSISTANT,
            )
            return [system_message, assistant_message]
        else:
            return None

    def _predict(self, model_input: MessageHistory) -> str:
        user_requests = model_input.role_based_history(role=Role.USER)
        return self._llm_model(user_requests)


class CallCustomerNode(MultifunctionNode):
    def greeting_message(self) -> Optional[MessageOutput]:
        message_history = MessageHistory(messages=[])
        message_history.add_user_message(
            content=f"Call user on his phone number: {self._node_input.phone_number}"
        )
        completion = self._predict(message_history)
        if self._output_parser is not None:
            ticket_request: PhoneCallTicket = self._output_parser.parse(completion)
            return MessageOutput(
                message=f"\nThanks for your time today with {ticket_request.agent_name}  a ticket has been created on your behalf"
                f"\nHere is your ticket summary: "
                f"\n\n{ticket_request.call_summary}"
                f"\n\nThanks for your time today! See you next time",
                role=Role.ASSISTANT,
            )
        return None

    def no_edges_found(self, user_input: NodeInput) -> Optional[MessageOutput]:
        return None

    def _get_tools(self):
        tools = [
            Tool.from_function(
                func=call_customer,
                description="Use to call premium customers",
                name="call_customer_tool",
                return_direct=True,
            )
        ]
        return tools
