from langchain_chroma import Chroma

def fetch_relevant_answer(user_input: str, vector_store: Chroma):
    results = vector_store.similarity_search(user_input, k=1)  
    if results:
        return results[0].page_content
    else:
        return "No relevant documents found."
