from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import os
import torch
from transformers import AutoModel, AutoTokenizer,BitsAndBytesConfig

load_dotenv()

def load_finetuned_bert():
    model = any
    tokenizer =any
    bnb_config = BitsAndBytesConfig( load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModel.from_pretrained( "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1", quantization_config=bnb_config, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained("ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1", trust_remote_code=True)
    print("✅ load_finetuned_bert:", model)
    return model, tokenizer

def load_finetuned_model():
    llm = HuggingFaceEndpoint(
        repo_id="sudeelmez/health_qa_model_llama3",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_KEY"],
        temperature=0.7,
        max_new_tokens=512,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    print("✅ load_finetuned_model:", llm)
    return llm



def init_rag():
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_ollama import OllamaEmbeddings

    urls = [
        "https://ada.com/editorial/reducing-risk-type-2-diabetes/",
        "https://ada.com/editorial/deal-with-covid-and-skin-issues/",
        "https://ada.com/editorial/how-to-lower-cholesterol/",
        "https://ada.com/editorial/5-healthy-habits-for-2024/",
        "https://ada.com/editorial/what-does-my-headache-mean/",
        "https://ada.com/editorial/how-to-care-for-skin-with-psoriasis/"
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    embeddings = OllamaEmbeddings(model="llama3.1")
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
    )

    return vectorstore.as_retriever(k=6)

def init_llms():
    from langchain_ollama import ChatOllama

    llm_deepseek = ChatOllama(model="deepseek-coder:6.7b")
    llm_llama = ChatOllama(model="llama3.1")
    
    print("✅ llm_deepseek:", llm_deepseek)
    print("✅ llm_llama:", llm_llama)

    return llm_deepseek, llm_llama
