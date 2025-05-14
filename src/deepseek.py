from openai import OpenAI
import re
import os
import threading
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_AI_API_KEY"),
)

def run_with_timeout(func, args=(), timeout=15):
    result = [None]
    
    def wrapper():
        try:
            result[0] = func(*args)
        except Exception as e:
            result[0] = f"Error: {str(e)}"
    
    thread = threading.Thread(target=wrapper)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print(f"🔴 [ERROR] Yanıt alınırken hata oluştu: Yanıt süresi aşıldı...")
        return ""
    
    return result[0]

def get_deepseek_answer(user_input: str) -> str:
    print(f"\n🟡 [INFO] Kullanıcı girdisi alındı:\n{user_input}\n")

    def fetch_completion():
        return client.chat.completions.create(
            extra_body={},
            model="deepseek/deepseek-r1-zero:free",
            messages=[
                {
                    "role": "system",
                    "content": "Sen kullanıcı sorularına doğal ve açık bir şekilde yanıt veren bir asistansın. Cevaplarını açık ve düzenli bir şekilde ver."
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        )

    completion = run_with_timeout(fetch_completion, timeout=20)

    if isinstance(completion, str) and completion.startswith("Error:"):
        print(f"🔴 [ERROR] {completion}")
        return completion

    try:
        raw_answer = completion.choices[0].message.content
    except Exception as e:
        print(f"🔴 [ERROR] Yanıt alınırken hata oluştu: {str(e)}")
        return ""

    # print(f"\n🟢 [RAW] Modelden gelen cevap:\n{raw_answer}")

    match = re.search(r"\\boxed\{(.*?)\}", raw_answer, re.DOTALL)
    cleaned_answer = match.group(1).strip() if match else raw_answer.strip()

    # print(f"\n✅ [CLEANED] Temizlenmiş cevap:\n{cleaned_answer}\n")
    return cleaned_answer
