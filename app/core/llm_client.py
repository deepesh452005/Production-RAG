"""
Wraps the Groq API. Rest of the app never calls Groq directly —
swap providers later by only touching this file.
"""
import yaml
from pathlib import Path
from groq import Groq
from app.config import settings

_client = Groq(api_key=settings.groq_api_key)

_PROMPTS_PATH = Path("prompts.yaml")
_prompts = yaml.safe_load(_PROMPTS_PATH.read_text(encoding="utf-8"))


def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = _prompts["system_prompt"]
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    response = _client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

def rewrite_query(question: str) -> str:
    prompt = _prompts["query_rewrite_prompt"].format(question=question)

    response = _client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    rewritten = response.choices[0].message.content.strip()
    return rewritten if rewritten else question  # fallback: never return an empty query