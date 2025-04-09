from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

# from openai import OpenAI

# client = OpenAI(
#     base_url = 'http://localhost:11434/v1',
#     api_key='ollama', # required, but unused
# )

# completion = client.chat.completions.create(
# extra_body={},
# model="deepseek/deepseek-r1-zero:free",
# messages=[
#     {
#       "role": "user",
#       "content": "nasılsın? bugün istanbulda hava nasıl"
#     }
#   ]
#   )


# print(completion.choices[0].message.content)


import re

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

