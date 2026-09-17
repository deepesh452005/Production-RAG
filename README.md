# Production-Grade Retrieval-Augmented Generation (RAG) Pipeline

An end-to-end RAG system built from scratch, with a focus on production-level retrieval quality, automated evaluation, and real-world usability — not just a working demo.

## Overview

This project answers questions over a curated knowledge base (space exploration, cryptography, internet history, and ocean exploration) using a retrieval pipeline that combines semantic search, keyword search, and cross-encoder re-ranking, guarded against hallucination, and served through a simple chat interface.

Every architectural decision in this project was made to solve a specific, observed problem — not copied from a tutorial. See the **Design Decisions & Findings** section below for the reasoning and evidence behind each one.

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| API framework | FastAPI | Async, auto-generated docs, lightweight |
| Pipeline orchestration | LangGraph | Explicit graph structure with real conditional branches (guardrail, cache routing) |
| Vector database | ChromaDB | Simple, local, persistent |
| Embeddings | ChromaDB's built-in ONNX model (`all-MiniLM-L6-v2`) | Avoids a `torch` dependency entirely |
| Keyword search | `rank_bm25` (BM25Okapi) | Pure algorithm, no model, catches exact terms embeddings miss |
| Re-ranker | FlashRank (`ms-marco-MiniLM-L-12-v2`) | Cross-encoder precision without a `torch` dependency |
| LLM | Groq (`openai/gpt-oss-20b`) | Fast, free-tier friendly |
| Document formats | `.txt`, `.docx` | Real-world document support |
| Evaluation | Custom LLM-as-judge + RAGAS | Two independent evaluation frameworks |

## Architecture

```
Question
   │
   ▼
[Rewrite] ── LLM cleans up casual/typo'd phrasing into a clear question
   │
   ▼
[Cache Check] ── keyed on the rewritten query; skips everything below on a hit
   │
   ▼
[Retrieve] ── Hybrid search: ChromaDB (vector) + BM25 (keyword), fused via
   │           Reciprocal Rank Fusion
   ▼
[Re-rank] ── FlashRank cross-encoder scores each candidate against the
   │          original question; keeps the top 4
   ▼
[Guardrail] ── if the best score is below threshold → refuse
   │                                              │
   ▼ (passes)                                     ▼
[Generate] ── LLM answers using only              [Refuse] ── "I don't have
   the retrieved context                           enough information..."
   │                                              │
   └──────────────────┬───────────────────────────┘
                       ▼
                 [Save to Cache]
                       ▼
                     [Log]
```

Built as an explicit LangGraph state graph — the guardrail and cache-check are real conditional edges, not buried `if` statements, making the pipeline's structure visible and easy to extend (e.g., with agent-style steps later).

## Key Features

- **Hybrid retrieval** — combines semantic (vector) and keyword (BM25) search so both paraphrased questions and exact terms/names/dates are matched well
- **Cross-encoder re-ranking** — a second, more precise pass that resolves ambiguity between similar-looking chunks (e.g., correctly distinguishing adjacent facts like "first human in space" vs "first American in space")
- **Anti-hallucination guardrail** — refuses to answer, before ever calling the LLM, when retrieval confidence is too low — a code-enforced guarantee, not a hoped-for model behavior
- **Query rewriting** — normalizes casual, vague, or typo'd questions into clean queries before retrieval
- **Smart caching** — caches by the *rewritten* query (not the raw question), so typo'd or reworded repeats still hit the cache and skip redundant LLM calls
- **Config-driven prompts** — all prompts live in `prompts.yaml`, tunable without touching code
- **Structured logging** — every request logged as JSONL (question, answer, retrieved chunks, scores, latency) for debugging
- **A working chat UI** — plain HTML/CSS/JS, served directly by FastAPI, no extra dependencies

## Evaluation

Two independent, automated evaluation frameworks — both function as quality gates (non-zero exit code on failure), suitable for CI-style regression checking.

### Custom LLM-as-judge (`eval/run_eval.py`)
Scores retrieval accuracy and answer correctness against an 18-question test set spanning exact facts, disambiguation, paraphrase, and out-of-scope refusal cases.

**Latest result:** 17/18 (94%) overall pass rate, 18/18 (100%) retrieval pass rate.

### RAGAS (`eval/ragas_eval.py`)
Industry-standard metrics: faithfulness, answer relevancy, context precision, context recall.

**Latest result:**
| Metric | Score |
|---|---|
| Faithfulness | 1.00 |
| Answer Relevancy | 0.97 |
| Context Precision | 0.99 |
| Context Recall | 1.00 |

## Ablation Study — Proving the Upgrades Actually Helped

`eval/ablation_study.py` isolates the retrieval-accuracy contribution of each upgrade stage, using a harder test set (16 questions built around adjacent, easy-to-confuse facts) across a larger, more realistic 4-document corpus.

| Stage | Retrieval Accuracy |
|---|---|
| Semantic Chunking (vector search only) | 88% |
| + Hybrid Search (vector + BM25) | 100% |
| + Reranking | 100% |

**Finding:** semantic chunking alone missed 2 questions where a compound fact (e.g., "Challenger" + "1986") was split across adjacent chunks. Hybrid search's keyword matching fully recovered both cases, and re-ranking maintained that reliability — demonstrating that the combination of techniques, not any single one in isolation, is what delivers robust retrieval.

## Project Structure

```
rag-project/
├── app/
│   ├── main.py                    # FastAPI app, serves the API + chat UI
│   ├── config.py                  # Centralized settings (.env-based)
│   ├── core/
│   │   ├── rag_service.py         # LangGraph pipeline definition
│   │   ├── llm_client.py          # Groq wrapper (generate + rewrite)
│   │   └── cache.py                # Response caching
│   ├── ingestion/
│   │   ├── loader.py               # .txt / .docx document loader
│   │   └── chunker.py              # Recursive, boundary-aware chunking
│   ├── retrieval/
│   │   ├── vector_store.py         # ChromaDB wrapper
│   │   ├── keyword_store.py        # BM25 wrapper
│   │   ├── hybrid_retriever.py     # RRF fusion of both
│   │   └── reranker.py             # FlashRank cross-encoder
│   └── utils/
│       └── logger.py               # JSONL request logging
├── eval/
│   ├── test_set.json                # Custom eval question set
│   ├── judge.py                     # LLM-as-judge grader
│   ├── run_eval.py                  # Custom eval runner + quality gate
│   ├── ragas_eval.py                # RAGAS eval runner + quality gate
│   ├── ablation_study.py            # Retrieval-accuracy ablation harness
│   └── ablation_docs/               # Larger corpus used only for ablation testing
├── static/
│   └── index.html                   # Chat UI
├── data/                             # Production documents + persisted indexes
├── prompts.yaml                      # All LLM prompts (editable without code changes)
├── ingest.py                         # One-time ingestion script
└── requirements.txt
```

## Setup

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate   # Windows
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set your Groq API key in `.env`:
   ```
   GROQ_API_KEY=your_key_here
   ```
4. Ingest the documents:
   ```
   python ingest.py
   ```
5. Run the server:
   ```
   uvicorn app.main:app --reload
   ```
6. Open `http://localhost:8000/` for the chat UI, or `http://localhost:8000/docs` for the API.

## Running Evaluations

```
python -m eval.run_eval          # Custom LLM-as-judge eval
python -m eval.ragas_eval        # RAGAS eval
python -m eval.ablation_study    # Retrieval-accuracy ablation study
```

## Design Decisions & Findings

A few notable engineering decisions made along the way, based on real debugging and testing rather than assumptions:

- **Avoided `torch` entirely** (embeddings via ChromaDB's ONNX model, re-ranking via FlashRank) after hitting a Windows long-path installation failure with `sentence-transformers`.
- **Recursive, boundary-aware chunking** replaced naive fixed-size chunking after observing that mid-sentence cuts degraded retrieval and generation quality.
- **Re-ranking uses the rewritten query, not the original**, after finding that FlashRank's cross-encoder scores dropped significantly on casual/vague phrasing — while retrieval itself benefited from the query rewrite.
- **Guardrail threshold tuned from 0.3 to 0.15** after query rewriting was added, since even correct answers to casually-phrased questions scored somewhat lower with the re-ranker; this was re-validated against the full eval suite (including refusal cases) to confirm no regression.
- **Caching keyed on the rewritten query, not the raw question** — catches typo'd and reworded duplicate questions that would otherwise miss an exact-match cache.

## Possible Next Steps

- Semantic caching (catching reworded duplicates without needing an LLM rewrite first)
- Multi-step retrieval for genuinely multi-part questions
- Agent capabilities (the LangGraph structure is already suited for this)
