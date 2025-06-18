from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from src.utils import State
from src.agents import (
    analyze_message,
    use_rag,
    evaluate_with_langsmith,
    generate_final_response,
    use_llama
)
from langsmith.run_trees import RunTree
from langchain_core.messages import BaseMessage

class SmartChatBot:
    def __init__(self, llm_llama, retriever):
        self.llm_llama = llm_llama
        self.retriever = retriever

        self.memory = MemorySaver()
        self.general_store = InMemoryStore()

        self.pipeline = RunTree(name="Smart Chat Pipeline", run_type="chain", inputs={"question": ""})
        self.pipeline.post()
        self.graph_builder = StateGraph(State)
        self._build_graph()

    def run_chat_session(self, user_input: str, conv_id: str = "chat-001") -> str:
        state = {
            "messages": [{"role": "user", "content": user_input}]
        }
        retriever = self.retriever 
        config = {"configurable": {
            "thread_id": conv_id, 
            "llama_llm": self.llm_llama,
            "retriever": retriever
            }}

        self.pipeline.inputs["question"] = user_input
        self.pipeline.patch()

        final_message = ""

        for event in self.graph.stream(state, config):
            for value in event.values():
                if value is not None:
                    try:
                        last_msg = value["messages"][-1]
                        if isinstance(last_msg, BaseMessage):
                            final_message = last_msg.content
                        else:
                            final_message = last_msg["content"]
                    except Exception as e:
                        final_message = f"⚠️ Error: {str(e)}"
        
        self.pipeline.end(outputs={"answer": final_message})
        self.pipeline.patch()

        return final_message

    def _build_graph(self):
        self.graph_builder.add_node("analyze_message", analyze_message)
        self.graph_builder.add_node("use_rag", use_rag)
        self.graph_builder.add_node("evaluate", evaluate_with_langsmith)
        self.graph_builder.add_node("generate_final_response", generate_final_response)
        self.graph_builder.add_node("use_llama", use_llama)

        self.graph_builder.add_edge(START, "analyze_message")
        self.graph_builder.add_edge("analyze_message", "use_llama")
        self.graph_builder.add_edge("analyze_message", "use_rag")
        
        self.graph_builder.add_edge("use_rag", "evaluate")
        self.graph_builder.add_edge("use_llama", "evaluate")
 
        self.graph_builder.add_edge("evaluate", "generate_final_response")
        self.graph_builder.add_edge("generate_final_response", END)

        self.graph = self.graph_builder.compile(checkpointer=self.memory)

    def stream_graph_updates(self, user_input: str):
        state = {
            "messages": [{"role": "user", "content": user_input}]
        }
        config = {"configurable": {
            "thread_id": "chat-001", 
            "llama_llm": self.llm_llama,
            "retriever": self.retriever}}
        self.pipeline.inputs["question"] = user_input
        self.pipeline.patch()

        for event in self.graph.stream(state, config):
            for value in event.values():
                if value is not None:
                    try:
                        last_msg = value["messages"][-1]
                        if isinstance(last_msg, BaseMessage):
                            message = last_msg.content
                        else:
                            message = last_msg["content"]
                    except Exception as e:
                        print(f"⚠️ Error accessing message content: {e}")
        self.pipeline.end(outputs={"answer": state["messages"][-1]["content"]})
        self.pipeline.patch()
