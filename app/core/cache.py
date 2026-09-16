"""
Exact-match response cache. If the exact same question has been asked
before, skip retrieval + generation entirely and return the saved answer.

Stored as a simple JSON file — good enough for a single-instance app.
A real multi-user production system would use Redis instead, but the
interface here (get/set) stays the same either way, so swapping later
is a small change.
"""
import json
import hashlib
from pathlib import Path

CACHE_PATH = Path("data/response_cache.json")


def _normalize(question: str) -> str:
    """Lowercase + strip whitespace, so 'What is X?' and 'what is x? ' hit the same entry."""
    return question.strip().lower()


def _hash_key(question: str) -> str:
    return hashlib.sha256(_normalize(question).encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def get_cached_response(question: str) -> dict | None:
    cache = _load_cache()
    return cache.get(_hash_key(question))


def set_cached_response(question: str, response: dict):
    cache = _load_cache()
    cache[_hash_key(question)] = response
    _save_cache(cache)