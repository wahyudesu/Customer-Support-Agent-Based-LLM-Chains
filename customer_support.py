from typing import Optional, List, Tuple

from langchain.chat_models import ChatOpenAI

from agents.support import UserInfoChainBasedEdge, AuthenticatedUserNode, GreetingNode, \
    CallCustomerEdge, CallCustomerNode
from data.chat import MessageHistory, Role
from data.graph import MessageOutput, EdgeOutput
from data.validation import UserProfile, PhoneCallTicket
from graph.node import BaseNode


class CustomerSupportPipeline:

    def __init__(self):
        #gpt-3.5-turbo
        self._llm_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        self._message_history = MessageHistory([])
        self._current_node = None

    def _get_pipeline(self) -> BaseNode:
        self._call_customer_node = CallCustomerNode(llm_model=self._llm_model,
                                                    pydantic_object=PhoneCallTicket,
                                                    edges=[],
                                                    final_state=True)
        self._call_customer_edge = CallCustomerEdge(llm_model=self._llm_model, out_node=self._call_customer_node)

        self._help_node = AuthenticatedUserNode(llm_model=self._llm_model,
                                                pydantic_object=None,
                                                edges=[self._call_customer_edge])

        self._user_info_chain = UserInfoChainBasedEdge(model=self._llm_model,
                                                       pydantic_object=UserProfile,
                                                       out_node=self._help_node)

        self._start_node = GreetingNode(edges=[self._user_info_chain])
        return self._start_node

    def _set_current_node(self, node: BaseNode) -> MessageOutput:
        self._current_node = node
        return node.greeting_message()

    def run(self, user_input: Optional[str]) -> Tuple[List[MessageOutput], bool]:
        if user_input is not None and user_input != "":
            self._message_history.add_user_message(content=user_input)

        assistant_output: List[MessageOutput] = []

        if self._current_node is None:
            greeting = self._set_current_node(self._get_pipeline())
            self._message_history.add_message(
                content=greeting.message,
                role=greeting.role
            )
            return [greeting], self._current_node.is_node_final()

        else:
            output = self._current_node.execute(self._message_history)
            if isinstance(output, EdgeOutput):
                if output.message_output is not None:
                    for msg_output in output.message_output:
                        self._message_history.add_message(content=msg_output.message, role=msg_output.role)
                        if msg_output.role == Role.ASSISTANT:
                            assistant_output.append(msg_output)

                if output.next_node is not None:
                    node_output = self._set_current_node(output.next_node)
                    if isinstance(node_output, MessageOutput):
                        self._message_history.add_assistant_message(content=node_output.message)

                    if node_output.role == Role.ASSISTANT:
                        assistant_output.append(node_output)

            elif isinstance(output, MessageOutput):
                self._message_history.add_message(content=output.message, role=output.role)
                if output.role == Role.ASSISTANT:
                    assistant_output.append(output)

            return assistant_output, self._current_node.is_node_final()


if __name__ == "__main__":
    def print_messages(res):
        if res is not None:
            for out in res:
                if isinstance(out, MessageOutput):
                    print(out.message)

    pipeline = CustomerSupportPipeline()
    res, is_over = pipeline.run("")
    print_messages(res)

    while not is_over:
        query = input()
        res, is_over = pipeline.run(query)
        print_messages(res)
