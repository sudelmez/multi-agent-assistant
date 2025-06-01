from src.chat import SmartChatBot
from src.init import init_rag, init_llms
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, Request
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
    allow_origins=["*"]
)

try:
    logger.info("🔧 RAG başlatılıyor...")
    retriever = init_rag()
    logger.info("✅ RAG başlatıldı.")

    logger.info("🧠 LLM'ler başlatılıyor...")
    llm_deepseek, llm_llama = init_llms()
    logger.info("✅ LLM'ler başlatıldı.")

    logger.info("🤖 Chatbot oluşturuluyor...")
    chatbot = SmartChatBot(llm_llama=llm_llama, retriever=retriever)
    logger.info("✅ Chatbot hazır.")

except Exception as e:
    logger.error(f"🚨 Başlatma hatası: {e}")
    chatbot = None

chatbot = SmartChatBot(llm_llama=llm_llama, retriever=retriever)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if chatbot is None:
        logger.warning("❌ Chatbot henüz başlatılamadı.")
        return ChatResponse(response="Chatbot şu anda kullanılamıyor.")
    logger.info(f"📨 Kullanıcı mesajı alındı: {request.message}")
    response = chatbot.run_chat_session(request.message)
    logger.info(f"💬 Bot yanıtı: {response}")
    return ChatResponse(response=response)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)