"""
Logs every RAG request as one JSON object per line (JSONL format).
Makes it easy to debug: was a bad answer a retrieval problem
(wrong chunks) or a generation problem (right chunks, bad answer)?
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "rag_requests.jsonl"


def log_request(question: str, retrieved: list[dict], answer: str, elapsed_seconds: float):
    LOG_DIR.mkdir(exist_ok=True)

    entry = {
        "question": question,
        "answer": answer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "retrieved": [
            {
                "source": chunk["source"],
                "rerank_score": chunk["rerank_score"],
                "text_preview": chunk["text"][:150],
            }
            for chunk in retrieved
        ],
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")