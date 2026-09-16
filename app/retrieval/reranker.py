"""
Re-ranks hybrid search's shortlist using a cross-encoder model.
Unlike vector/BM25 search (which score chunks independently and fast),
a cross-encoder reads the query and each chunk TOGETHER, catching
nuance that similarity/keyword scoring alone misses (e.g. distinguishing
"SC-4041" from "SC-5102" even when both chunks look textually similar).

Uses FlashRank (onnxruntime-based) instead of sentence-transformers'
CrossEncoder, to avoid the torch dependency entirely.
"""
from flashrank import Ranker, RerankRequest

_ranker = None

def _get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _ranker


def rerank(query: str, candidates: list[dict], top_k: int = 4) -> list[dict]:
    """
    candidates: list of {"text", "source", ...} from hybrid_retriever.
    Returns the top_k candidates, reordered by cross-encoder relevance score.
    """
    if not candidates:
        return []

    passages = [
        {"id": i, "text": c["text"], "meta": {"source": c["source"]}}
        for i, c in enumerate(candidates)
    ]

    request = RerankRequest(query=query, passages=passages)
    results = _get_ranker().rerank(request)  # already sorted, highest relevance first

    reranked = []
    for r in results[:top_k]:
        original = candidates[r["id"]]
        reranked.append({
            "text": original["text"],
            "source": original["source"],
            "rerank_score": round(float(r["score"]), 4),
        })
    return reranked