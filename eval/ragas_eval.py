"""
Phase 3, upgraded: evaluation using RAGAS instead of a custom LLM-judge.

RAGAS gives four separately-measured signals instead of one blended
pass/fail:
  - faithfulness:       is the answer grounded in the retrieved context?
  - answer_relevancy:   does the answer actually address the question?
  - context_precision:  were the retrieved chunks actually useful?
  - context_recall:     did retrieval find everything needed to answer?

Uses Groq (via langchain-groq) as the judge LLM, and FastEmbed (onnxruntime,
no torch) for the embedding-based metrics — keeping the same no-torch
philosophy as the rest of this project.

Run with:
    python -m eval.ragas_eval
"""
import json
import math
import sys
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, AnswerRelevancy, context_precision, context_recall
answer_relevancy = AnswerRelevancy(strictness=1)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from eval.safe_embeddings import SafeFastEmbedEmbeddings

from app.config import settings
from app.core.rag_service import ask

TEST_SET_PATH = Path("eval/test_set.json")
RESULTS_PATH = Path("eval/ragas_results.json")

# Thresholds — tune these as you learn what's realistic for your system
MIN_FAITHFULNESS = 0.7
MIN_ANSWER_RELEVANCY = 0.7
MIN_CONTEXT_PRECISION = 0.6
MIN_CONTEXT_RECALL = 0.6

judge_llm = LangchainLLMWrapper(
    ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.0,
        max_tokens=4096,
      reasoning_effort="low"
    )
)
judge_embeddings = LangchainEmbeddingsWrapper(SafeFastEmbedEmbeddings())


def build_ragas_dataset(test_cases: list[dict]) -> Dataset:
    """
    Runs each question through the actual pipeline and assembles the
    format RAGAS expects: question, answer, contexts (list of strings),
    ground_truth (reference answer).

    Out-of-scope ("REFUSE") cases are skipped here — RAGAS's metrics are
    designed to score real answers against real context, not refusals.
    The guardrail's refusal behavior is already covered by run_eval.py.
    """
    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for case in test_cases:
        if case["expected_answer"] == "REFUSE":
            continue

        response = ask(case["question"])
        contexts = [chunk["text"] for chunk in response.get("retrieved", [])]

        records["question"].append(case["question"])
        records["answer"].append(response["answer"])
        records["contexts"].append(contexts if contexts else [""])
        records["ground_truth"].append(case["expected_answer"])

    return Dataset.from_dict(records)


def run_ragas_eval():
    test_cases = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))

    print(f"Running {len(test_cases)} test cases through the pipeline "
          f"(skipping out-of-scope cases for RAGAS scoring)...\n")

    dataset = build_ragas_dataset(test_cases)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config= RunConfig(max_workers=1),
        raise_exceptions=True,
    )

    scores = result.to_pandas()
    scores.to_json(RESULTS_PATH, orient="records", indent=2)

    avg_faithfulness = scores["faithfulness"].mean()
    avg_relevancy = scores["answer_relevancy"].mean()
    avg_precision = scores["context_precision"].mean()
    avg_recall = scores["context_recall"].mean()

    print("\n" + "=" * 50)
    print(f"Faithfulness:       {avg_faithfulness:.2f}  (min required: {MIN_FAITHFULNESS})")
    print(f"Answer Relevancy:   {avg_relevancy:.2f}  (min required: {MIN_ANSWER_RELEVANCY})")
    print(f"Context Precision:  {avg_precision:.2f}  (min required: {MIN_CONTEXT_PRECISION})")
    print(f"Context Recall:     {avg_recall:.2f}  (min required: {MIN_CONTEXT_RECALL})")
    print("=" * 50)
    print(f"\nFull per-question results written to {RESULTS_PATH}")

    scores_map = {
        "faithfulness": avg_faithfulness,
        "answer_relevancy": avg_relevancy,
        "context_precision": avg_precision,
        "context_recall": avg_recall,
    }
    nan_metrics = [name for name, val in scores_map.items() if math.isnan(val)]
    if nan_metrics:
        print(f"\nWARNING: these metrics failed to compute (NaN): {nan_metrics}")

    failed = (
        bool(nan_metrics)
        or avg_faithfulness < MIN_FAITHFULNESS
        or avg_relevancy < MIN_ANSWER_RELEVANCY
        or avg_precision < MIN_CONTEXT_PRECISION
        or avg_recall < MIN_CONTEXT_RECALL
    )

    if failed:
        print("\nQUALITY GATE FAILED — one or more RAGAS metrics below threshold.")
        sys.exit(1)
    else:
        print("\nQuality gate passed.")
        sys.exit(0)


if __name__ == "__main__":
    run_ragas_eval()