import asyncio
from uuid import uuid4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
import fitz
import os

class RagItemHelpers:
    def __init__(self, vector_store, embeddings):

        self.vector_store = vector_store
        self.embeddings = embeddings

        self.urls  = [
            "https://hepiyi.com.tr/yardim-ve-destek/urunler-hakkinda/kasko-sigortasi",
            # "https://hepiyi.com.tr/yardim-ve-destek/urunler-hakkinda/trafik-sigortasi",
            # "https://hepiyi.com.tr/yardim-ve-destek/urunler-hakkinda/tamamlayici-saglik-sigortasi",
            # "https://hepiyi.com.tr/yardim-ve-destek/urunler-hakkinda/konut-sigortasi",
            # "https://hepiyi.com.tr/yardim-ve-destek/urunler-hakkinda/dask",
            # "https://hepiyi.com.tr/yardim-ve-destek/urunler-hakkinda/alternatif-kasko"
        ]
        self.pdf_files = ["./data/covid.pdf", "./data/anemia.pdf", "./data/cholera.pdf", "./data/covid.pdf", "./data/health.pdf"]

    async def scrape_page_metadata(self, url):
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                title = soup.title.string if soup.title else "No title"
                description_tag = soup.find("meta", {"name": "description"})
                description = description_tag["content"] if description_tag else "No description available"
                return title, description
            else:
                print(f"Failed to fetch URL {url}, status code: {response.status_code}")
                return None, None

    async def upload_web_content_to_chroma(self, urls):
        print("Uploading web content to Chroma...")
        for url in urls:
            if not self.vector_store.get_documents({"source": url}):
                loader = WebBaseLoader(web_paths=[url])
                data = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
                all_splits = text_splitter.split_documents(data)
                title, description = await self.scrape_page_metadata(url)
                
                if all_splits:
                    for split in all_splits:
                        page_content = split.page_content
                        metadata = {
                            "source": url,
                            "title": title,  
                            "description": description, 
                            "language": "tr"
                        }
                        document = Document(
                            page_content=page_content,
                            metadata=metadata
                        )
                        self.vector_store.add_documents(
                            documents=[document],
                            ids=[str(uuid4())],
                        )
                        print(f"Uploaded content from {url}")
            else:
                print(f"Content from {url} already exists in the database.")

    def read_pdf(self, file_path):
        text = ""
        with fitz.open(file_path) as pdf_file:
            for page in pdf_file:
                text += page.get_text()
        return text            

    async def upload_pdfs_to_chroma(self, pdf_files):
        print("Uploading PDF documents to Chroma...")

        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                print(f"File {pdf_file} not found, skipping.")
                continue

            try:
                text = self.read_pdf(pdf_file)

                if not text.strip():
                    print(f"No text found in {pdf_file}, skipping.")
                    continue

                document = Document(
                    page_content=text,
                    metadata={"source": pdf_file}
                )

                self.vector_store.add_documents(
                    documents=[document],
                    ids=[str(uuid4())],
                )
                print(f"Uploaded PDF document {pdf_file}")
            except Exception as e:
                print(f"Failed to upload {pdf_file}: {e}")

    # async def upload_pdfs_to_chroma(self, pdf_files):
    #     print("Uploading PDF documents to Chroma...")
    #     for pdf_file in pdf_files:
    #         if not self.vector_store.get_documents({"source": pdf_file}):
    #             text = self.read_pdf(pdf_file)

    #             document = Document(
    #                 page_content=text,
    #                 metadata={"source": pdf_file}
    #             )

    #             self.vector_store.add_documents(
    #                 documents=[document],
    #                 ids=[str(uuid4())],
    #             )
    #             print(f"Uploaded PDF document {pdf_file}")
    #         else:
    #             print(f"PDF document {pdf_file} already exists in the database.")

    async def upload_rag_items(self):
        try:
            await asyncio.gather(
                # self.upload_web_content_to_chroma(urls=self.urls),
                self.upload_pdfs_to_chroma(self.pdf_files)
            )
            print("Web content and PDF content uploaded successfully.")
        except Exception as e:
            print(f"Error during web content or PDF upload: {e}")
