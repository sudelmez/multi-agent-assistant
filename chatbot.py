from typing import Annotated
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
import uuid

class State(TypedDict):
    messages: Annotated[list, add_messages]

class ChatBot:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.1")
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        
        self.graph_builder = StateGraph(State)
        self._build_graph()

    def _build_graph(self):
        self.graph_builder.add_node("analyze_message", self.analyze_message)
        self.graph_builder.add_node("chatbot", self.chatbot)
        self.graph_builder.add_node("generate_final_response", self.generate_final_response)

        self.graph_builder.add_edge(START, "analyze_message")
        self.graph_builder.add_edge("analyze_message", "chatbot")
        self.graph_builder.add_edge("chatbot", "generate_final_response")
        self.graph_builder.add_edge("generate_final_response", END)

        self.graph = self.graph_builder.compile(checkpointer=self.checkpointer, store=self.store)

    def analyze_message(self, state: State):
        user_input = state["messages"][-1].content
        if "weather" in user_input.lower():
            new_message = {"role": "user", "content": f"Please provide the current weather information, {user_input}"}
        else:
            new_message = {"role": "user", "content": f"{user_input}"}
        
        state["messages"][-1] = new_message
        return state 

    def chatbot(self, state: State):
        response = self.llm.invoke(state["messages"])
        chatbot_message = {"role": "assistant", "content": str(response)} 
        state["messages"][-1] = chatbot_message
        return state

    def generate_final_response(self, state: State):
        search_result = state["messages"][-1].content
        final_response = {"role": "assistant", "content": f"Based on the query '{search_result}', the result is: {search_result}"}
        state["messages"][-1] = final_response
        return state  

    def stream_graph_updates(self, user_input: str):
        state = {"messages": [{"role": "user", "content": user_input}]}
        config = {"configurable": {"thread_id": "1"}}
        
        for event in self.graph.stream(state, config):
            for value in event.values():
                if value is not None:
                    print("Assistant:", value["messages"][-1]["content"])
                else:
                    print("Error: Received None, skipping...")