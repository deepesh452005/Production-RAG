"""
Combines vector_store.py (semantic) and keyword_store.py (BM25) results
into a single ranked list. This is the "hybrid" in hybrid search.

Approach: reciprocal rank fusion (RRF) — a simple, well-established way
to merge two differently-scaled ranking systems (distance vs. BM25 score)
without needing to normalize or weight them by hand. Each result's final
score is based on *where it ranked* in each list, not the raw numbers.

Note: ChromaDB persists itself to disk automatically. BM25 does not —
it's an in-memory index — so chunks are also saved to a JSON file here.
That lets a freshly-started server process rebuild the same keyword
index that ingest.py built, without re-running ingestion.
"""
import json
from pathlib import Path
from app.retrieval.vector_store import VectorStore
from app.retrieval.keyword_store import KeywordStore

RRF_K = 60
CHUNKS_CACHE_PATH = Path("data/chunks_cache.json")


class HybridRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.keyword_store = KeywordStore()
        self._load_keyword_index_if_available()

    def _load_keyword_index_if_available(self):
        if CHUNKS_CACHE_PATH.exists():
            chunks = json.loads(CHUNKS_CACHE_PATH.read_text(encoding="utf-8"))
            self.keyword_store.index(chunks)

    def index(self, chunks: list[dict]):
        self.vector_store.add_chunks(chunks)
        self.keyword_store.index(chunks)

        CHUNKS_CACHE_PATH.parent.mkdir(exist_ok=True)
        CHUNKS_CACHE_PATH.write_text(json.dumps(chunks), encoding="utf-8")

    def query(self, query_text: str, top_k: int = 4, return_pool: int = None) -> list[dict]:
        # Pull a slightly larger candidate pool from each source, then fuse
        candidate_pool = max(top_k * 2, 8)
        trim_to = return_pool or top_k

        vector_hits = self.vector_store.query(query_text, top_k=candidate_pool)
        keyword_hits = self.keyword_store.query(query_text, top_k=candidate_pool)

        rrf_scores: dict[str, float] = {}
        chunk_lookup: dict[str, dict] = {}

        for rank, hit in enumerate(vector_hits):
            chunk_id = f"{hit['source']}::{hit['text'][:30]}"
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (RRF_K + rank + 1)
            chunk_lookup[chunk_id] = hit

        for rank, hit in enumerate(keyword_hits):
            chunk_id = f"{hit['source']}::{hit['text'][:30]}"
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (RRF_K + rank + 1)
            chunk_lookup[chunk_id] = hit

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:trim_to]

        results = []
        for chunk_id in ranked_ids:
            hit = chunk_lookup[chunk_id]
            results.append({
                "text": hit["text"],
                "source": hit["source"],
                "hybrid_score": round(rrf_scores[chunk_id], 4),
            })
        return results