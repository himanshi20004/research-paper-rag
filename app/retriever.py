from app.ingestor import get_vectorstore

def retrieve_chunks(query: str, k: int = 4) -> list:
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return docs