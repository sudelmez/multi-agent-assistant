from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel
from typing import List
from PIL import Image
from PyPDF2 import PdfReader
import pytesseract
import base64
from io import BytesIO
import logging
import uvicorn

from src.chat import SmartChatBot
from src.init import init_rag, init_llms

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
    conversationId: str

class ChatResponse(BaseModel):
    response: str

class UserDocRequest(BaseModel):
    user_id: str
    documents: List[str]

class UserDocResponse(BaseModel):
    success: bool
    message: str
user_vectorstores = {} 
user_vectorstore = None

def decode_base64_file(base64_str: str) -> bytes:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    return base64.b64decode(base64_str)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def extract_text_from_image(image_bytes: bytes) -> str:
    image = Image.open(BytesIO(image_bytes))
    return pytesseract.image_to_string(image)

def extract_text_from_file(base64_file: str) -> str:
    file_bytes = decode_base64_file(base64_file)

    if file_bytes[0:4] == b'%PDF':
        return extract_text_from_pdf(file_bytes)
    elif file_bytes[0:2] == b'\xff\xd8' or file_bytes[0:8] == b'\x89PNG\r\n\x1a\n':
        return extract_text_from_image(file_bytes)
    else:
        raise ValueError("Desteklenmeyen dosya türü")

@app.post("/upload-user-data", response_model=UserDocResponse)
async def upload_user_data(request: UserDocRequest):
    try:
        logger.info(f"📤 Kullanıcı verisi alındı: {request.user_id} - {len(request.documents)} belge")

        texts = []
        for base64_file in request.documents:
            try:
                text = extract_text_from_file(base64_file)
                if text.strip():
                    texts.append(text)
            except Exception as e:
                logger.warning(f"📄 Dosya işlenemedi: {e}")

        if not texts:
            raise ValueError("Geçerli metin içeren dosya bulunamadı.")

        docs = [Document(page_content=t, metadata={"user_id": request.user_id, "source": "user_data"}) for t in texts]
        logger.info(f"📄 {len(docs)} adet Document nesnesi oluşturuldu.")

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
        doc_chunks = text_splitter.split_documents(docs)
        logger.info(f"✂️ {len(doc_chunks)} belge parçası oluşturuldu.")

        embeddings = OllamaEmbeddings(model="llama3.1")
        global user_vectorstore
        user_vectorstore = InMemoryVectorStore.from_documents(doc_chunks, embedding=embeddings)
        user_vectorstores[request.user_id] = user_vectorstore
        logger.info(f"✅ {request.user_id} için belgeler başarıyla vektör veritabanına yüklendi.")
        return UserDocResponse(success=True, message="Dosyalar başarıyla yüklendi.")

    except Exception as e:
        logger.error(f"🚨 Dosya yükleme hatası: {e}")
        raise HTTPException(status_code=500, detail="Dosyalar yüklenirken hata oluştu.")

    
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_id = "1"
    user_retriever = user_vectorstores.get(user_id)
    if chatbot is None:
        logger.warning("❌ Chatbot henüz başlatılamadı.")
        return ChatResponse(response="Chatbot şu anda kullanılamıyor.")
    
    from langchain.retrievers import EnsembleRetriever

    combined_retriever = EnsembleRetriever(
        retrievers=[retriever, user_retriever.as_retriever()],
        weights=[0.5, 1.0]  
    )

    logger.info(f"📨 Kullanıcı mesajı alındı: {request.message}")
    session_bot = SmartChatBot(llm_llama=llm_llama, retriever=combined_retriever)
    response = session_bot.run_chat_session(request.message, request.conversationId)

    logger.info(f"💬 Bot yanıtı: {response}")
    return ChatResponse(response=response)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)