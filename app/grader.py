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
    temperature=0
)

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", """Grade ALL these document chunks at once for relevance to the question.
    For each chunk, reply with just 'yes' or 'no' on a new line, in order.
    No explanations. Just yes/no for each chunk."""),
    ("human", "Question: {question}\n\nChunks:\n{chunks}")
])

grader_chain = grader_prompt | llm | StrOutputParser()


def grade_documents(query: str, docs: list) -> tuple[list, bool]:
    if not docs:
        return [], True

    # Grade ALL chunks in ONE single API call instead of one per chunk
    chunks_text = "\n\n---\n\n".join([
        f"Chunk {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])

    time.sleep(2)  # small delay to respect rate limits

    result = grader_chain.invoke({
        "question": query,
        "chunks": chunks_text
    })

    lines = [l.strip().lower() for l in result.strip().split("\n") if l.strip()]

    relevant_docs = []
    any_irrelevant = False

    for i, doc in enumerate(docs):
        verdict = lines[i] if i < len(lines) else "no"
        if "yes" in verdict:
            relevant_docs.append(doc)
        else:
            any_irrelevant = True

    return relevant_docs, any_irrelevant