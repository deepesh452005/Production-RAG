"""
Uses an LLM as a judge to score whether a generated answer matches the
meaning of an expected answer. This avoids brittle exact-string matching
("175 billion" vs "175 billion parameters" should both count as correct).

This is a standard technique in RAG evaluation: since natural language
has many correct phrasings, a small, focused LLM call judges semantic
correctness instead of comparing strings character-by-character.
"""
import json
from groq import Groq
from app.config import settings

_client = Groq(api_key=settings.groq_api_key)

JUDGE_PROMPT = """You are grading whether a generated answer correctly conveys \
the same information as an expected answer.

Question: {question}
Expected answer: {expected}
Generated answer: {generated}

Respond with ONLY a JSON object, no other text, in this exact format:
{{"correct": true or false, "reason": "one short sentence explaining why"}}

A generated answer is "correct" if it conveys the same key facts as the \
expected answer, even if worded differently. It is "incorrect" if it is \
missing key facts, contradicts the expected answer, or is vague/evasive \
when specific facts were expected."""


def judge_answer(question: str, expected: str, generated: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, generated=generated)

    response = _client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic grading
    )

    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # If the judge didn't return clean JSON, fail safe rather than crash the eval run
        return {"correct": False, "reason": f"Judge returned unparseable output: {raw[:100]}"}