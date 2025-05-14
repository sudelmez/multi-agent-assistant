from chat import SmartChatBot
from init import init_rag, init_llms, load_finetuned_model, load_finetuned_bert
from dotenv import load_dotenv

load_dotenv()

retriever = init_rag() 
llm_deepseek, llm_llama = init_llms()  
# finetuned_model = load_finetuned_model()
# finetuned_bert, tokenizer = load_finetuned_bert()

chatbot = SmartChatBot(llm_llama=llm_llama, retriever=retriever)

while True:
    user_input = input("👤 You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("🛑 Exiting chatbot...")
        break
    chatbot.stream_graph_updates(user_input)
