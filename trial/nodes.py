from typing import Annotated
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

checkpointer = InMemorySaver()
store = InMemoryStore()
from helpers import extract_name

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
llm = ChatOllama(model="llama3.1")


def analyze_message(state: State):
    user_input = state["messages"][-1].content

    if "weather" in user_input.lower():
        new_message = {"role": "user", "content": f"Please provide the current weather information, {user_input}"}
    else:
        new_message = {"role": "user", "content": f"{user_input}"}
    
    state["messages"][-1] = new_message
    return state 

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    chatbot_message = {"role": "assistant", "content": str(response)} 
    state["messages"][-1] = chatbot_message

    return state

def generate_final_response(state: State):
    search_result = state["messages"][-1].content
    final_response = {"role": "assistant", "content": f"Based on the query '{search_result}', the result is: {search_result}"}
    state["messages"][-1] = final_response

    return state  

graph_builder.add_node("analyze_message", analyze_message)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("generate_final_response", generate_final_response)

graph_builder.add_edge(START, "analyze_message")
graph_builder.add_edge("analyze_message", "chatbot")
graph_builder.add_edge("chatbot", "generate_final_response")
graph_builder.add_edge("generate_final_response", END)

graph = graph_builder.compile(checkpointer=checkpointer, store=store)

config = {"configurable": {"thread_id": "1"}}
def stream_graph_updates(user_input: str):
    state = {"messages": [{"role": "user", "content": user_input}]}
    for event in graph.stream(state, config):
        for value in event.values():
            if value is not None:
                print("Assistant:", value["messages"][-1]["content"])
            else:
                print("Error: Received None, skipping...")

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        print(f"Error: {e}")
        user_input = "What do you know about LangGraph?"
        stream_graph_updates(user_input)
        break
