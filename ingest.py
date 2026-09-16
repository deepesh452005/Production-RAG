"""
Run this whenever you add/update documents in data/.
    python ingest.py
"""
from app.retrieval.hybrid_retriever import HybridRetriever
from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_text
from app.retrieval.vector_store import VectorStore


DATA_DIR = "data"

def main():
    print(f"Loading documents from {DATA_DIR}/ ...")
    docs = load_documents(DATA_DIR)
    print(f"  Found {len(docs)} document(s).")

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"], doc["source"])
        all_chunks.extend(chunks)
    print(f"  Split into {len(all_chunks)} chunk(s).")

    retriever = HybridRetriever()
    retriever.index(all_chunks)
    print(f"Done. Indexed {len(all_chunks)} chunk(s) into vector store and keyword index.")


if __name__ == "__main__":
    main()
