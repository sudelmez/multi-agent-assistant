from chatbot import ChatBot

if __name__ == "__main__":
    chatbot_instance = ChatBot()
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
