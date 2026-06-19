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

# One single prompt that returns BOTH scores together — saves 1 API call
eval_prompt = ChatPromptTemplate.from_messages([
    ("system", """Evaluate the answer on two criteria. Reply with ONLY two numbers on two lines:
Line 1: Faithfulness score (0-10) — does the answer stick to the context without hallucinating?
Line 2: Relevance score (0-10) — does the answer address the question?

Only two numbers. Nothing else."""),
    ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}")
])

eval_chain = eval_prompt | llm | StrOutputParser()


def evaluate_answer(question: str, answer: str, context_docs: list) -> dict:
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    time.sleep(2)  # respect rate limits

    try:
        result = eval_chain.invoke({
            "question": question,
            "context": context_text[:2000],  # trim to avoid token limits
            "answer": answer
        })

        lines = [l.strip() for l in result.strip().split("\n") if l.strip()]
        faith_score = int(''.join(filter(str.isdigit, lines[0]))) if len(lines) > 0 else 5
        rel_score   = int(''.join(filter(str.isdigit, lines[1]))) if len(lines) > 1 else 5

        # clamp to 0-10
        faith_score = max(0, min(10, faith_score))
        rel_score   = max(0, min(10, rel_score))

    except Exception as e:
        print(f"Eval error: {e}")
        faith_score, rel_score = 5, 5

    confidence = round((faith_score + rel_score) / 2 * 10)

    # Log to LangSmith
    try:
        from langsmith import Client
        langsmith_client = Client()
        langsmith_client.create_example(
            inputs={"question": question},
            outputs={
                "answer": answer,
                "faithfulness": faith_score,
                "relevance": rel_score,
                "confidence": confidence
            },
            dataset_name="research-rag-evals"
        )
    except Exception as e:
        print(f"LangSmith logging skipped: {e}")

    return {
        "faithfulness": faith_score,
        "relevance": rel_score,
        "confidence": confidence
    }