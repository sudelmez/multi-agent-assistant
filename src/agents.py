from src.utils import get_message_content, State
from src.rag import rag_bot
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

@traceable()
def analyze_message(state: State):
    user_message = get_message_content(state["messages"][-1])
    state.setdefault("process", {})["analyzed_user_message"] = user_message
    return state

@traceable()
def use_rag(state: State, config: dict):
    user_input = state["process"]["analyzed_user_message"]
    llama_llm = config.get("configurable", {}).get("llama_llm")
    retriever = config.get("configurable", {}).get("retriever")

    rag_result = rag_bot(user_input, llama_llm, retriever)

    state["process"]["rag_result"] = rag_result["answer"]
    state["process"]["rag_docs"] = rag_result["documents"]
    return state

@traceable()
def use_llama(state: State, config: dict):
    llama_llm = config.get("configurable", {}).get("llama_llm")
    llama_result = llama_llm.invoke(state["messages"])

    state["process"]["llama_result"] = llama_result
    return state

@traceable()
def evaluate_with_langsmith(state: State):
    answers = {
        "rag": state["process"].get("rag_result"),
        "llama": state["process"].get("llama_result")
    }
    filtered_answers = [
        {"name": name, "content": content}
        for name, content in answers.items()
        if content
    ]
    return state

@traceable()
def generate_final_response(state: State, config: dict):
    process = state["process"]
    rag_answer = process.get("rag_result", "")
    llama_answer = process.get("llama_result", "")

    llama_llm = config.get("configurable", {}).get("llama_llm")
    
    final_prompt = f"""
    You are a licensed medical doctor responding to a patient. You were given two assistant answers below:

    - The first is from a general assistant with memory of the conversation.
    - The second is from a health expert assistant with access to medical knowledge.

    Please write a **single, clear, medically accurate** response that:

    1. Combines the most helpful parts of both responses.
    2. Resolves contradictions in favor of the Health Expert Assistant’s response.
    3. Uses professional, caring, and empathetic tone—like a doctor advising a patient.
    4. Avoids meta language such as “Here’s the final answer” or “According to...” and speaks directly as a doctor would.
    5. If the Health Expert included a confidence level, append it in this format:  
    **(Medical information confidence level: XX%)**

    Respond as if you are the doctor speaking to the patient directly.

    ---

    General Assistant Response:
    {llama_answer}

    ---

    Health Expert Assistant Response:
    {rag_answer}

    ---

    Doctor's Response:
    """

    final_result = llama_llm.invoke(final_prompt)

    state["messages"].append({
        "role": "assistant",
        "content": final_result.content
    })

    print("\n---\nAssistant:", final_result.content)
    return state
