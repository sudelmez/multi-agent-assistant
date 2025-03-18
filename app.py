from chatbot import ChatBot
from rag_helpers import RagItemHelpers
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
import asyncio
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

async def main():
    embeddings = OllamaEmbeddings(model="llama3.1")
    llm = ChatOllama(model="llama3.1")

    vector_store = Chroma(
        collection_name="rag_items",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  
    )
    
    helpers = RagItemHelpers(vector_store=vector_store, embeddings=embeddings)
    # await helpers.upload_rag_items()

    chatbot_instance = ChatBot(llm=llm, vector_store=vector_store)

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            chatbot_instance.stream_graph_updates(user_input)
        except Exception as e:
            print(f"Error: {e}")
            user_input = "What do you know about LangGraph?"
            chatbot_instance.stream_graph_updates(user_input)
            break

if __name__ == "__main__":
    asyncio.run(main())