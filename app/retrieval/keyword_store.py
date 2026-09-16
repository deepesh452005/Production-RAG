"""
Keyword-based (BM25) search. Complements vector_store.py's semantic search —
BM25 excels at exact term/keyword matches (IDs, names, jargon) that
embeddings can miss, since it scores based on word overlap, not meaning.
"""
from rank_bm25 import BM25Okapi


class KeywordStore:
    def __init__(self):
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[dict]):
        self._chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def query(self, query_text: str, top_k: int = 4) -> list[dict]:
        if not self._bm25 or not self._chunks:
            return []

        tokenized_query = query_text.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self._chunks, scores), key=lambda pair: pair[1], reverse=True
        )[:top_k]

        return [
            {"text": c["text"], "source": c["source"], "chunk_id": c["chunk_id"], "score": float(score)}
            for c, score in ranked
            if score > 0
        ]