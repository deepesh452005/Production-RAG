"""
Wraps ChromaDB so the rest of the app never touches Chroma's API directly.
This matters for production: if we swap ChromaDB for Qdrant later,
only this file changes.
"""
import chromadb
from chromadb.utils import embedding_functions
from app.config import settings


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

        # ChromaDB's built-in ONNX-based embedding function — same all-MiniLM-L6-v2
        # model as sentence-transformers, but uses onnxruntime instead of torch.
        # Avoids torch entirely (no long-path install issues on Windows).
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=self.embed_fn,
        )

    def add_chunks(self, chunks: list[dict]):
        """chunks: list of {"text", "source", "chunk_id"}"""
        if not chunks:
            return
        self.collection.add(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"]} for c in chunks],
        )

    def query(self, query_text: str, top_k: int = None) -> list[dict]:
        """Returns top_k most similar chunks with their source + distance."""
        k = top_k or settings.top_k
        results = self.collection.query(query_texts=[query_text], n_results=k)

        hits = []
        for i in range(len(results["documents"][0])):
            hits.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i],
            })
        return hits

    def count(self) -> int:
        return self.collection.count()
