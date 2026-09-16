"""
Phase 3 evaluation script.

Runs every question in eval/test_set.json through the actual RAG pipeline
(app.core.rag_service.ask), scores each response, and prints a report.

Exits with code 1 (failure) if the pass rate is below MIN_PASS_RATE —
this is what makes it usable as a CI-style quality gate: a real "build
fails" signal, not just a report you have to read and remember to check.

Run with:
    python -m eval.run_eval
"""
import json
import sys
from pathlib import Path

from app.core.rag_service import ask
from eval.judge import judge_answer

TEST_SET_PATH = Path("eval/test_set.json")
RESULTS_PATH = Path("eval/last_run_results.json")

MIN_PASS_RATE = 0.8      # overall: 80% of questions must pass
MIN_RETRIEVAL_RATE = 0.8  # retrieval alone must also hit 80%


def check_retrieval(retrieved: list[dict], expected_keywords: list[str]) -> bool:
    """Did any retrieved chunk contain at least one expected keyword?"""
    if not expected_keywords:
        return True  # out-of-scope questions have no expected keywords
    combined_text = " ".join(chunk["text"].lower() for chunk in retrieved)
    return any(keyword.lower() in combined_text for keyword in expected_keywords)


def run_eval():
    test_cases = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    results = []

    print(f"Running {len(test_cases)} test cases...\n")

    for case in test_cases:
        question = case["question"]
        expected = case["expected_answer"]
        expected_keywords = case["expected_source_keywords"]

        response = ask(question)
        generated_answer = response["answer"]
        retrieved = response.get("retrieved", [])
        guardrail_fired = response.get("guardrail_triggered", False)

        retrieval_ok = check_retrieval(retrieved, expected_keywords)

        if expected == "REFUSE":
            answer_ok = guardrail_fired
            judge_reason = "guardrail fired as expected" if guardrail_fired else "guardrail did NOT fire, but should have"
        else:
            verdict = judge_answer(question, expected, generated_answer)
            answer_ok = verdict["correct"]
            judge_reason = verdict["reason"]

        passed = answer_ok and (retrieval_ok or expected == "REFUSE")

        results.append({
            "id": case["id"],
            "type": case["type"],
            "question": question,
            "generated_answer": generated_answer,
            "expected_answer": expected,
            "retrieval_ok": retrieval_ok,
            "answer_ok": answer_ok,
            "passed": passed,
            "judge_reason": judge_reason,
        })

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {case['id']} ({case['type']}): {question}")
        if not passed:
            print(f"       -> {judge_reason}")

    # --- Aggregate scoring ---
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    retrieval_passed_count = sum(1 for r in results if r["retrieval_ok"])

    pass_rate = passed_count / total
    retrieval_rate = retrieval_passed_count / total

    print("\n" + "=" * 50)
    print(f"Overall pass rate:    {passed_count}/{total} ({pass_rate:.0%})")
    print(f"Retrieval pass rate:  {retrieval_passed_count}/{total} ({retrieval_rate:.0%})")
    print("=" * 50)

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull results written to {RESULTS_PATH}")

    # --- Quality gate ---
    if pass_rate < MIN_PASS_RATE or retrieval_rate < MIN_RETRIEVAL_RATE:
        print(f"\nQUALITY GATE FAILED (need >= {MIN_PASS_RATE:.0%} pass rate, "
              f">= {MIN_RETRIEVAL_RATE:.0%} retrieval rate)")
        sys.exit(1)
    else:
        print(f"\nQuality gate passed.")
        sys.exit(0)


if __name__ == "__main__":
    run_eval()