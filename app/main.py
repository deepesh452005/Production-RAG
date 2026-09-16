from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from app.core.rag_service import ask
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.core.rag_service import ask
from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_text
from app.core.rag_service import ask, get_retriever

app = FastAPI(title="RAG Assistant - Phase 1")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def ensure_ingested():
    marker = Path("data/chunks_cache.json")
    if not marker.exists():
        print("No existing index found — running ingestion...")
        docs = load_documents("data")
        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunk_text(doc["text"], doc["source"]))
        print(f"Found {len(docs)} document(s), split into {len(all_chunks)} chunk(s).")
        get_retriever().index(all_chunks)
        print("Ingestion complete.")

@app.get("/")
def serve_chat_ui():
    return FileResponse("static/index.html")
class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(payload: Question):
    return ask(payload.question)
