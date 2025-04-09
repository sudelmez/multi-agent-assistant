from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv
from langsmith.run_trees import RunTree
from deepseek import get_deepseek_answer
from llama3 import get_llama_answer

load_dotenv() 
class State(TypedDict):
    messages: Annotated[list, add_messages]

class ChatBot:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        
        self.pipeline = RunTree(
            name="Chat Pipeline",
            run_type="chain",
            inputs={"question": ""}  
        )
        self.pipeline.post()

        self.graph_builder = StateGraph(State)
        self._build_graph()

    def _build_graph(self):
        self.graph_builder.add_node("analyze_message", self.analyze_message)
        self.graph_builder.add_node("use_rag", self.use_rag)
        self.graph_builder.add_node("ask_llama", self.ask_llama)
        self.graph_builder.add_node("ask_deepseek", self.ask_deepseek)
        self.graph_builder.add_node("ai_process", self.ai_process)
        self.graph_builder.add_node("generate_final_response", self.generate_final_response)

        self.graph_builder.add_edge(START, "analyze_message")
        self.graph_builder.add_edge("analyze_message", "use_rag")
        self.graph_builder.add_edge("use_rag", "ai_process")
        self.graph_builder.add_edge("ai_process", "generate_final_response")
        # self.graph_builder.add_edge("use_rag", "ask_deepseek")
        # self.graph_builder.add_edge("ask_deepseek", "ask_llama")
        # self.graph_builder.add_edge("ask_llama", "generate_final_response")
        self.graph_builder.add_edge("generate_final_response", END)

        self.graph = self.graph_builder.compile(checkpointer=self.checkpointer, store=self.store)

    def analyze_message(self, state: State):
        user_input = state["messages"][-1].content
        if "weather" in user_input.lower():
            new_message = {"role": "user", "content": f"Please provide the current weather information, {user_input}"}
        else:
            new_message = {"role": "user", "content": f"{user_input}"}
        
        self.pipeline.create_child(
            name="Analyze Message",
            run_type="tool",
            inputs={"user_input": user_input}
        ).post()

        state["messages"][-1] = new_message
        return state 
    
    def use_rag(self, state: State):
        print("\n---\nrag state stopped...")
        # user_input = state["messages"][-1].content

        # relevant_docs = self.vector_store.similarity_search_with_score(user_input, k=1)

        # if relevant_docs:
        #     context = "\n".join([doc.page_content for doc, _ in relevant_docs])
        #     system_message = {"role": "system", "content": f"Use the following context to answer:\n{context}"}
        # else:
        # system_message = {
        #             "role": "system",
        #             "content": f"{question}"
        #         }
        # state["messages"].append(system_message)

        question = self.pipeline.inputs["question"]
        system_message = {
            "role": "system",
            "content": f"{question}"
        }
        state["messages"].append(system_message)

        self.pipeline.create_child(
            name="Rag Call",
            run_type="llm", 
            inputs={"messages": state["messages"]}
        ).post()

        return state

    def ai_process(self, state: State):
        last_message = state["messages"][-1]
        user_input = last_message["content"] if isinstance(last_message, dict) else last_message.content

        deepseek_response = get_llama_answer(user_input, self.llm)
        llama_response = get_llama_answer(user_input, self.llm)

        deepseek_message = {"role": "assistant", "content": deepseek_response}
        llama_message = {"role": "assistant", "content": str(llama_response)}

        combined_content = f"İki LLM modelinin verdiği cevaplar:\n\nDeepSeek: {deepseek_message}\n\nLLaMA: {llama_message}"
        state["messages"].append({"role": "assistant", "content": combined_content})

        print("\n\n-----------\nCombined Content\n", combined_content, "---------")

        self.pipeline.create_child(
            name="Ai Process Call",
            run_type="llm",
            inputs={"messages": state["messages"]}
        ).post()

        return state

    def ask_deepseek(self, state: State):
        last_message = state["messages"][-1]
        user_input = last_message["content"] if isinstance(last_message, dict) else last_message.content
        deepseek_response = get_deepseek_answer(user_input)
        deepseek_message = {"role": "assistant", "content": deepseek_response}

        state["messages"].append(deepseek_message)

        self.pipeline.create_child(
            name="Deepseek Call",
            run_type="llm", 
            inputs={"messages": state["messages"]}
        ).post()

        return state
    
    def ask_llama(self, state: State):
        last_message = state["messages"][-1]
        user_input = last_message["content"] if isinstance(last_message, dict) else last_message.content
        response = get_llama_answer(user_input, self.llm)

        chatbot_message = {"role": "assistant", "content": str(response)} 

        state["messages"][-1] = chatbot_message

        self.pipeline.create_child(
            name="Chatbot Call",
            run_type="llm", 
            inputs={"messages": state["messages"]}
        ).post()

        return state

    def generate_final_response(self, state: State):
        search_result = state["messages"][-1].content
        final_response = {"role": "assistant", "content": f"Based on the query '{search_result}', the result is: {search_result}"}
        state["messages"][-1] = final_response

        self.pipeline.create_child(
            name="Generate Final Response",
            run_type="tool",
            inputs={"search_result": search_result}
        ).post()

        return state  


    def stream_graph_updates(self, user_input: str):
        state = {"messages": [{"role": "user", "content": user_input}]}
        config = {"configurable": {"thread_id": "1"}}

        self.pipeline.inputs["question"] = user_input
        self.pipeline.patch()
        
        for event in self.graph.stream(state, config):
            for value in event.values():
                if value is not None:
                    print("\n---\nAssistant:", value["messages"][-1]["content"])
                else:
                    print("\n---\nError: Received None, skipping...")
        
        self.pipeline.end(outputs={"answer": state["messages"][-1]["content"]})
        self.pipeline.patch()