"""Presentation metrics: retrieval accuracy (free) + latency/token percentiles (real Groq cost).

    python scripts/eval_metrics.py                 # retrieval accuracy only, zero cost
    python scripts/eval_metrics.py --with-latency  # also runs real generations

Retrieval accuracy is pure vector search, so it costs nothing and can be run as
often as you like. --with-latency issues one real Groq call per question, paced to
stay inside the free-tier rate limit.

Questions live in eval_questions.json so the set can grow without touching code.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.chain import run_rag
from app.rag.retrieval import select_context_docs

QUESTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "eval_questions.json")


def load_questions():
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def check_retrieval_accuracy(test_questions):
    results = []
    correct = 0
    for tq in test_questions:
        docs, _scored = select_context_docs(tq["question"])
        retrieved_ids = sorted({d.metadata.get("arxiv_id") for d in docs})
        hit = any(e in retrieved_ids for e in tq["expected"])
        correct += hit
        results.append({
            "question": tq["question"],
            "expected": tq["expected"],
            "retrieved": retrieved_ids,
            "hit": hit,
        })
    accuracy = correct / len(test_questions) * 100 if test_questions else 0.0
    return accuracy, results


def measure_latency_and_usage(test_questions, delay_seconds=1.5):
    """delay_seconds paces requests so a ~40-question batch doesn't trip the
    Groq free-tier rate limit."""
    latencies, usages = [], []
    for tq in test_questions:
        result = run_rag(tq["question"], debug=False)
        latencies.append(result["latency_ms"])
        usages.append(result["usage"])
        time.sleep(delay_seconds)
    return latencies, usages


def percentile(data, pct):
    data = sorted(data)
    idx = min(int(len(data) * pct), len(data) - 1)
    return data[idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-latency", action="store_true",
                        help="also run real generations to measure latency and tokens")
    args = parser.parse_args()

    questions = load_questions()

    accuracy, results = check_retrieval_accuracy(questions)
    print(f"Retrieval accuracy: {accuracy:.1f}% "
          f"({sum(r['hit'] for r in results)}/{len(results)})\n")
    for r in results:
        if not r["hit"]:
            print("MISS:", r)

    if not args.with_latency:
        return

    print("\nRunning real generations (this spends Groq tokens)...")
    latencies, usages = measure_latency_and_usage(questions)

    print(f"\np50 latency: {percentile(latencies, 0.50)} ms")
    print(f"p95 latency: {percentile(latencies, 0.95)} ms")
    print(f"min/max: {min(latencies)} / {max(latencies)} ms\n")

    total_tokens = sum(u.get("total_tokens", 0) for u in usages)
    total_calls = sum(u.get("llm_calls", 0) for u in usages)
    print(f"Total LLM calls across {len(questions)} questions: {total_calls}")
    print(f"Total tokens: {total_tokens} (avg {total_tokens / len(questions):.0f}/question)")


if __name__ == "__main__":
    main()
