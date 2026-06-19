import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful research assistant. Answer the user's question 
    based ONLY on the provided context. 
    If the context doesn't contain enough information, say so clearly.
    Always cite which part of the context your answer comes from.
    Be concise but complete."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

answer_chain = answer_prompt | llm | StrOutputParser()


def generate_answer(question: str, context_docs: list, web_context: str = "") -> str:
    pdf_context = "\n\n".join([
        f"[PDF - Page {doc.metadata.get('page', '?')}]:\n{doc.page_content}"
        for doc in context_docs
    ])

    full_context = pdf_context
    if web_context:
        full_context += f"\n\n[WEB SEARCH RESULTS]:\n{web_context}"

    time.sleep(2)  # respect rate limits

    return answer_chain.invoke({
        "context": full_context[:3000],  # trim to avoid token limits
        "question": question
    })