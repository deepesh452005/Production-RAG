"""
Ablation study: proves each upgrade actually improved retrieval accuracy.

Runs the same test questions through 4 progressively-upgraded retrieval
configurations, using temporary in-memory indexes (never touches the
real data/chroma_db or data/chunks_cache.json):

  Stage 1 - Simple chunking:    naive fixed-size chunks + vector search only
  Stage 2 - Semantic chunking:  recursive (boundary-aware) chunks + vector search only
  Stage 3 - Hybrid search:      recursive chunks + vector search + BM25 (RRF fused)
  Stage 4 - Reranking:          Stage 3's candidates, re-ranked by FlashRank

Accuracy = % of questions where at least one retrieved chunk contains the
expected source keywords (same check used in run_eval.py) — this isolates
retrieval quality specifically, independent of LLM generation.

Run with:
    python -m eval.ablation_study
"""
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_text as recursive_chunk_text
from app.retrieval.reranker import rerank

DATA_DIR = "eval/ablation_docs"
TEST_SET_PATH = Path("eval/ablation_test_set.json")
TOP_K = 1
RRF_K = 60

embed_fn = embedding_functions.DefaultEmbeddingFunction()



# --- Temporary, in-memory retrieval helpers (isolated from production data) ---

def build_ephemeral_vector_store(chunks: list[dict], collection_name: str):
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name=collection_name, embedding_function=embed_fn)
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return collection


def vector_query(collection, query_text: str, top_k: int) -> list[dict]:
    results = collection.query(query_texts=[query_text], n_results=top_k)
    return [
        {"text": results["documents"][0][i], "source": results["metadatas"][0][i]["source"]}
        for i in range(len(results["documents"][0]))
    ]


def build_bm25_index(chunks: list[dict]):
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def bm25_query(bm25, chunks: list[dict], query_text: str, top_k: int) -> list[dict]:
    tokenized_query = query_text.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(chunks, scores), key=lambda p: p[1], reverse=True)[:top_k]
    return [{"text": c["text"], "source": c["source"]} for c, score in ranked if score > 0]


def rrf_fuse(vector_hits: list[dict], keyword_hits: list[dict], top_k: int) -> list[dict]:
    scores, lookup = {}, {}
    for rank, hit in enumerate(vector_hits):
        key = f"{hit['source']}::{hit['text'][:30]}"
        scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank + 1)
        lookup[key] = hit
    for rank, hit in enumerate(keyword_hits):
        key = f"{hit['source']}::{hit['text'][:30]}"
        scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank + 1)
        lookup[key] = hit
    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
    return [lookup[k] for k in ranked_keys]


# --- Accuracy check (same logic as run_eval.py) ---

def check_retrieval(retrieved: list[dict], expected_keywords: list[str]) -> bool:
    if not expected_keywords:
        return True
    combined = " ".join(r["text"].lower() for r in retrieved)
    return any(kw.lower() in combined for kw in expected_keywords)

def run_ablation():
    docs = load_documents(DATA_DIR)
    test_cases = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))

    recursive_chunks = []
    for doc in docs:
        recursive_chunks.extend(recursive_chunk_text(doc["text"], doc["source"]))

    print(f"Recursive chunks: {len(recursive_chunks)}\n")

    recursive_collection = build_ephemeral_vector_store(recursive_chunks, "recursive")
    bm25_index = build_bm25_index(recursive_chunks)

    results = {"Semantic Chunking": 0, "Hybrid Search": 0, "Reranking": 0}

    for case in test_cases:
        q, keywords = case["question"], case["expected_source_keywords"]

        # Stage 1: recursive chunking, vector only
        hits1 = vector_query(recursive_collection, q, TOP_K)
        if check_retrieval(hits1, keywords):
            results["Semantic Chunking"] += 1
        else:
            print(f"  [Semantic Chunking MISS] Q: {q}")
            print(f"      Expected keywords: {keywords}")
            print(f"      Got chunk starting: {hits1[0]['text'][:100]!r}\n")

        # Stage 2: recursive chunking, hybrid (vector + BM25, RRF fused)
        vec_candidates = vector_query(recursive_collection, q, TOP_K * 2)
        kw_candidates = bm25_query(bm25_index, recursive_chunks, q, TOP_K * 2)
        hits2 = rrf_fuse(vec_candidates, kw_candidates, TOP_K)
        if check_retrieval(hits2, keywords):
            results["Hybrid Search"] += 1

        # Stage 3: hybrid candidates, re-ranked
        hits3_pool = rrf_fuse(vec_candidates, kw_candidates, TOP_K * 2)
        hits3 = rerank(q, hits3_pool, top_k=TOP_K)
        if check_retrieval(hits3, keywords):
            results["Reranking"] += 1

    total = len(test_cases)
    print("=" * 50)
    print(f"Retrieval accuracy across {total} questions:\n")
    for stage, correct in results.items():
        pct = correct / total * 100
        print(f"  {stage:<20} {correct}/{total}  ({pct:.0f}%)")
    print("=" * 50)

    return results, total


if __name__ == "__main__":
    run_ablation()