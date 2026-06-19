import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
# CORRECT
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

FAISS_PATH = "./faiss_db"

def ingest_pdf(pdf_path: str) -> int:
    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    print("Creating embeddings (this takes ~30 seconds)...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(FAISS_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_PATH)
    print(f"Saved FAISS index to {FAISS_PATH}")

    return len(chunks)


def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )