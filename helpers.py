import re 

def extract_name(user_input: str) -> str:
    match = re.search(r"\bmy name is (\w+)\b", user_input, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""