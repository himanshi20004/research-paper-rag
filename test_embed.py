import os
from dotenv import load_dotenv
load_dotenv()

print("Testing FAISS + Gemini embedding...")
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

docs = [Document(page_content="This is a test sentence about research papers.")]
db = FAISS.from_documents(docs, embeddings)
results = db.similarity_search("test sentence", k=1)
print(f"FAISS works! Result: {results[0].page_content}")