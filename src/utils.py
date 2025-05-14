from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated, Optional, Any
import operator
from operator import or_ 

def get_message_content(message):
    try:
        if isinstance(message, dict):
            print(f"[DEBUG] message is dict: {message}")
            return message.get("content", "")
        elif hasattr(message, "content"):
            print(f"[DEBUG] message is object: {type(message)} with content")
            return message.content
        else:
            print(f"[DEBUG] message is something else: {message}")
            return str(message)
    except Exception as e:
        print("Error while getting message content:", e)
        return ""


class MessageProcess(TypedDict, total=False):
    analyzed_user_message: Optional[str]
    rag_result: Annotated[Optional[str], operator.add]
    rag_docs: Any
    llama_result: Annotated[Optional[str], operator.add]

class State(TypedDict):
    messages: Annotated[list, add_messages]
    process: Annotated[MessageProcess, or_]
