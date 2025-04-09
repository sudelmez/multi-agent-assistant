def get_llama_answer(user_input: str, llm: any):
    print("\n\n llama input....", user_input)
    response = llm.invoke(user_input)
    print("\n---\nllama state answer: ", response)
    return response.content