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
    analyzed_user_message: dict  

class ChatBot:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.general_memory = InMemorySaver()
        self.general_store = InMemoryStore()

        self.pipeline = RunTree(name="Chat Pipeline", run_type="chain", inputs={"question": ""})
        self.pipeline.post()

        self.graph_builder = StateGraph(State)
        self._build_graph()


    def get_message_content(self, message):
        return message["content"] if isinstance(message, dict) else getattr(message, "content", "")


    def _build_graph(self):
        self.graph_builder.add_node("analyze_message", self.analyze_message)
        self.graph_builder.add_node("use_rag", self.use_rag)
        self.graph_builder.add_node("ask_deepseek", self.ask_deepseek)
        self.graph_builder.add_node("ask_llama", self.ask_llama)
        self.graph_builder.add_node("generate_final_response", self.generate_final_response)

        self.graph_builder.add_edge(START, "analyze_message")
        self.graph_builder.add_edge("analyze_message", "use_rag")
        self.graph_builder.add_edge("use_rag", "ask_deepseek")
        self.graph_builder.add_edge("ask_deepseek", "ask_llama")
        self.graph_builder.add_edge("ask_llama", "generate_final_response")
        self.graph_builder.add_edge("generate_final_response", END)

        self.graph = self.graph_builder.compile(checkpointer=self.general_memory)

    def analyze_message(self, state: State):
        user_input = self.get_message_content(state["messages"][-1])
        if "weather" in user_input.lower():
            new_message = {"role": "user", "content": f"Please provide the current weather information, {user_input}"}
        else:
            new_message = {"role": "user", "content": user_input}
        
        self.pipeline.create_child(
            name="Analyze Message",
            run_type="tool",
            inputs={"user_input": user_input}
        ).post()

        state["messages"][-1] = new_message
        state["analyzed_user_message"] = new_message 

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
    

    def ask_deepseek(self, state: State):
        message = state.get("analyzed_user_message", state["messages"][-1])
        user_input = self.get_message_content(message)
        print("user_input on deepseek: ", user_input)
        response = get_deepseek_answer(user_input)
        state["messages"].append({"role": "assistant", "content": f"DeepSeek: {response}"})
        self.pipeline.create_child(name="DeepSeek Agent", run_type="llm", inputs={"input": user_input}).post()
        return state

    def ask_llama(self, state: State):
        message = state.get("analyzed_user_message", state["messages"][-1])
        user_input = self.get_message_content(message)  
        print("user_input on llama: ", user_input)
        response = get_llama_answer(user_input, self.llm)
        state["messages"].append({"role": "assistant", "content": f"LLaMA: {response}"})
        self.pipeline.create_child(name="LLaMA Agent", run_type="llm", inputs={"input": user_input}).post()
        return state

    def generate_final_response(self, state: State):
        deepseek_response = [self.get_message_content(msg) for msg in state["messages"] if "DeepSeek:" in self.get_message_content(msg)][-1]
        llama_response = [self.get_message_content(msg) for msg in state["messages"] if "LLaMA:" in self.get_message_content(msg)][-1]

        user_message = self.get_message_content(state.get("analyzed_user_message", state["messages"][0]))

        print("deepseek_response: ", deepseek_response)
        print("llama_response: ", llama_response)
        print("user_message: ", user_message)

        combined_input = (
            f"Kullanıcının sorusu: {user_message}\n\n"
            f"Gelen LLM cevaplarından çıkarımlar yaparak kullanıcıya en doğru ve anlamlı yanıtı oluştur. "
            f"Sen, bu cevapları analiz ederek nihai kararı veren bir ajansın. "
            f"Aynı kullanıcı ile konuşuyormuş gibi 1. kişi olarak cevap ver. Arkadaş canlısı yanıtlar üret."
            f"Kullanıcıya düzgün, açık ve anlaşılır bir yanıt oluştur:\n\n"
            f"DeepSeek cevabı: {deepseek_response}\n\n"
            f"LLaMA cevabı: {llama_response}"
        )

        print("combined_input: ", combined_input)

        final_response = get_llama_answer(combined_input, self.llm)
        state["messages"].append({"role": "assistant", "content": final_response})

        print("final_response: ", final_response)

        self.pipeline.create_child(
            name="Final Merge",
            run_type="llm",
            inputs={
                "input": combined_input,
                "user_question": user_message
            }
        ).post()

        return state


    def stream_graph_updates(self, user_input: str):
        state = {"messages": [{"role": "user", "content": user_input}]}
        config = {"configurable": {"thread_id": "12345"}}

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
