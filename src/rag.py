from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

@traceable()
def rag_bot(question: str, llm, retriever) -> dict:
    docs = retriever.invoke(question)

    user_docs = [doc for doc in docs if doc.metadata.get("source") == "user_data"]
    general_docs = [doc for doc in docs if doc.metadata.get("source") != "user_data"]

    general_docs_string = "\n\n".join(doc.page_content for doc in general_docs)
    user_docs_string = "\n\n".join(doc.page_content for doc in user_docs)
        
    full_context = f"""
        === GENERAL MEDICAL DOCUMENTS ===

        {general_docs_string}

        === USER-SPECIFIC MEDICAL DATA ===

        {user_docs_string}

        """

    instructions = f"""
        You are a highly specialized medical AI assistant trained to provide diagnostic insights and guidance using two sources:

        1. A trusted, high-quality medical knowledge base.
        2. Personal medical data specific to the user.

        **How to reason:**
        - Read the user's question carefully.
        - Use only the information provided in the knowledge base to formulate your response. Do not rely on any external knowledge or assumptions.
        - Use both the general documents and the user's personal data to answer the question. Do not rely on any external knowledge or assumptions.
        - Prioritize personal data when available — assume it reflects the user's current medical condition.
        - Do NOT guess. If there is not enough data to make a reasonable statement, clearly say so.
        - Be medically responsible, empathetic, and clear.

        Example closing:  
        "This assessment is based on your uploaded medical data and trusted documents. Likelihood of condition: [X]%."

        **Important Notes:**
        - The confidence score should reflect how strongly the evidence in the documents supports your answer. Give it as a float between 0.0 and 1.0, and present it as a percentage.
        - The user should understand clearly that the information comes from document-based analysis, not from general knowledge or guesswork.
        - Keep the explanation clear, empathetic, and medically responsible.

        **Question:**
        {question}

        **Context:**
        {full_context}
        """

    ai_msg = llm.invoke([
        {"role": "system", "content": instructions},
        {"role": "user", "content": question},
    ])

    return {"answer": ai_msg.content, "documents": docs}
