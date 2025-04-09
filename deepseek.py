from openai import OpenAI
import re
from dotenv import load_dotenv
import os

load_dotenv() 

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPEN_AI_API_KEY"),
)

def get_deepseek_answer(user_input: str):
    print("\n\ndeepseek input....", user_input)
    completion = client.chat.completions.create(
        extra_body={},
        model="deepseek/deepseek-r1-zero:free",
        messages=[
            {
                "role": "user",
                "content": user_input
            }
        ]
    )
    raw_answer = completion.choices[0].message.content
    match = re.search(r"\\boxed\{(.*?)\}", raw_answer, re.DOTALL)
    if match:
        cleaned_answer = match.group(1).strip()
    else:
        cleaned_answer = raw_answer.strip()  

    print("\n\ncleaned answer: ", cleaned_answer)
    return cleaned_answer

