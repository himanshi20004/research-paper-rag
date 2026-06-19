from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from app.retriever import retrieve_chunks
from app.grader import grade_documents
from app.web_search import web_search
from app.generator import generate_answer
from app.evaluator import evaluate_answer


# Define what our "state" looks like as it flows through the graph
class RAGState(TypedDict):
    question: str
    documents: List[Document]
    web_context: str
    answer: str
    used_web_search: bool
    scores: dict


# ── Node functions (each does one thing) ────────────────────────────────────

def retrieve(state: RAGState) -> RAGState:
    """Fetch chunks from ChromaDB."""
    docs = retrieve_chunks(state["question"])
    return {**state, "documents": docs}


def grade(state: RAGState) -> RAGState:
    """Grade each chunk. Flag if web search needed."""
    relevant_docs, needs_web = grade_documents(state["question"], state["documents"])
    return {
        **state,
        "documents": relevant_docs,
        "used_web_search": needs_web
    }


def search_web(state: RAGState) -> RAGState:
    """Run Tavily web search as a supplement."""
    web_result = web_search(state["question"])
    return {**state, "web_context": web_result}


def generate(state: RAGState) -> RAGState:
    """Generate the final answer."""
    answer = generate_answer(
        state["question"],
        state["documents"],
        state.get("web_context", "")
    )
    return {**state, "answer": answer}


def evaluate(state: RAGState) -> RAGState:
    """Score the answer and log to LangSmith."""
    scores = evaluate_answer(
        state["question"],
        state["answer"],
        state["documents"]
    )
    return {**state, "scores": scores}


# ── Routing logic ────────────────────────────────────────────────────────────

def should_search_web(state: RAGState) -> str:
    """After grading: go to web search if any chunks were irrelevant."""
    if state.get("used_web_search", False):
        return "web_search"
    return "generate"


# ── Build the graph ──────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(RAGState)

    # Add all nodes
    graph.add_node("retrieve",   retrieve)
    graph.add_node("grade",      grade)
    graph.add_node("web_search", search_web)
    graph.add_node("generate",   generate)
    graph.add_node("evaluate",   evaluate)

    # Set entry point
    graph.set_entry_point("retrieve")

    # Edges (flow)
    graph.add_edge("retrieve", "grade")

    # Conditional: after grading, go to web search OR directly to generate
    graph.add_conditional_edges("grade", should_search_web, {
        "web_search": "web_search",
        "generate":   "generate"
    })

    graph.add_edge("web_search", "generate")
    graph.add_edge("generate",   "evaluate")
    graph.add_edge("evaluate",   END)

    return graph.compile()


# Single compiled graph instance (reused across calls)
rag_graph = build_graph()


def run_query(question: str) -> dict:
    """Run the full corrective-RAG pipeline for a question."""
    initial_state: RAGState = {
        "question": question,
        "documents": [],
        "web_context": "",
        "answer": "",
        "used_web_search": False,
        "scores": {}
    }
    final_state = rag_graph.invoke(initial_state)
    return final_state