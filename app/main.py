from fastapi import FastAPI
from pydantic import BaseModel
from app.core.rag_service import ask
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.core.rag_service import ask

app = FastAPI(title="RAG Assistant - Phase 1")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def ensure_ingested():
    """
    If the vector database doesn't exist yet (e.g. first boot on a host
    with ephemeral storage), run ingestion automatically so the app
    always has a working knowledge base without manual setup.
    """
    chroma_path = Path("data/chroma_db")
    if not chroma_path.exists():
        print("No existing index found — running ingestion...")
        import subprocess
        subprocess.run(["python", "ingest.py"], check=True)
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
