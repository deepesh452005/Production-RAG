"""
The core RAG pipeline, built as an explicit LangGraph graph.

Caching is checked right after query rewriting (not before), so that
reworded/typo'd repeats of the same underlying question are recognized
as cache hits — the rewrite step already normalizes phrasing, so caching
on the rewritten query catches far more repeats than caching on the raw
question would.
"""
import time
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from app.core.cache import get_cached_response, set_cached_response
from app.config import settings
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import rerank
from app.core.llm_client import generate_answer, rewrite_query
from app.utils.logger import log_request

_retriever = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever

FINAL_TOP_K = 4
RERANK_CANDIDATE_POOL = 10
NO_CONTEXT_ANSWER = "I don't have enough information to answer that."


class RAGState(TypedDict, total=False):
    question: str
    rewritten_query: str
    start_time: float
    candidates: list[dict]
    hits: list[dict]
    answer: str
    guardrail_triggered: bool
    top_rerank_score: Optional[float]
    from_cache: bool


# --- Nodes ---

def rewrite_node(state: RAGState) -> dict:
    rewritten = rewrite_query(state["question"])
    print(f"\n[QUERY REWRITE] Original: {state['question']!r}  ->  Rewritten: {rewritten!r}\n")
    return {"rewritten_query": rewritten}


def cache_check_node(state: RAGState) -> dict:
    cached = get_cached_response(state["rewritten_query"])
    if cached is not None:
        return {
            "answer": cached["answer"],
            "hits": cached.get("retrieved", []),
            "guardrail_triggered": cached.get("guardrail_triggered", False),
            "top_rerank_score": cached.get("top_rerank_score"),
            "from_cache": True,
        }
    return {}


def retrieve_node(state: RAGState) -> dict:
    candidates = get_retriever().query(state["rewritten_query"], return_pool=RERANK_CANDIDATE_POOL)
    return {"candidates": candidates}


def rerank_node(state: RAGState) -> dict:
    hits = rerank(state["rewritten_query"], state["candidates"], top_k=FINAL_TOP_K)
    return {"hits": hits}


def generate_node(state: RAGState) -> dict:
    context_chunks = [h["text"] for h in state["hits"]]
    answer = generate_answer(state["question"], context_chunks)
    return {"answer": answer}


def refuse_node(state: RAGState) -> dict:
    top_score = state["hits"][0]["rerank_score"] if state["hits"] else None
    return {
        "answer": NO_CONTEXT_ANSWER,
        "guardrail_triggered": True,
        "top_rerank_score": top_score,
    }


def save_cache_node(state: RAGState) -> dict:
    response = {
        "answer": state["answer"],
        "retrieved": state.get("hits", []),
    }
    if state.get("guardrail_triggered"):
        response["guardrail_triggered"] = True
        response["top_rerank_score"] = state.get("top_rerank_score")

    set_cached_response(state["rewritten_query"], response)
    return {}


def log_node(state: RAGState) -> dict:
    elapsed = time.time() - state["start_time"]
    log_request(state["question"], state.get("hits", []), state["answer"], elapsed)
    return {}


# --- Conditional routing ---

def cache_router(state: RAGState) -> str:
    return "cache_hit" if state.get("from_cache") else "cache_miss"


def guardrail_router(state: RAGState) -> str:
    hits = state["hits"]
    if not hits:
        return "refuse"
    if hits[0]["rerank_score"] < settings.min_rerank_score:
        return "refuse"
    return "generate"


# --- Build the graph ---

def _build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("cache_check", cache_check_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("save_cache", save_cache_node)
    graph.add_node("log", log_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "cache_check")
    graph.add_conditional_edges(
        "cache_check",
        cache_router,
        {"cache_hit": "log", "cache_miss": "retrieve"},
    )
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank",
        guardrail_router,
        {"generate": "generate", "refuse": "refuse"},
    )
    graph.add_edge("generate", "save_cache")
    graph.add_edge("refuse", "save_cache")
    graph.add_edge("save_cache", "log")
    graph.add_edge("log", END)

    return graph.compile()


_pipeline = _build_graph()


def ask(question: str) -> dict:
    """
    Public entrypoint — same return shape as before, so main.py and the
    eval script don't need to change at all.
    """
    initial_state: RAGState = {"question": question, "start_time": time.time()}
    result = _pipeline.invoke(initial_state)

    response = {
        "answer": result["answer"],
        "retrieved": result.get("hits", []),
    }
    if result.get("guardrail_triggered"):
        response["guardrail_triggered"] = True
        response["top_rerank_score"] = result.get("top_rerank_score")
    if result.get("from_cache"):
        response["from_cache"] = True

    return response