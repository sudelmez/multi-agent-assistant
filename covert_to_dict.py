def convert_to_dict(msg):
    return {"role": msg.role, "content": msg.content} if hasattr(msg, "role") else msg
