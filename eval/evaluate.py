"""Retrieval quality evaluation for the Kubeflow docs-agent.

This script loads the golden dataset, runs each query through the
retrieval pipeline, and computes standard IR metrics:

- **Recall@k**: Fraction of expected sources found in the top-k results.
- **Precision@k**: Fraction of top-k results that match an expected source.
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for the first
  relevant result across all queries.

Usage::

    # Against a live Milvus instance:
    python -m eval.evaluate

    # With custom settings:
    python -m eval.evaluate --top-k 10 --dataset eval/golden_dataset.json

    # Dry-run (intent classification only, no Milvus required):
    python -m eval.evaluate --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_k(
    retrieved_sources: List[str],
    expected_patterns: List[str],
    k: int,
) -> float:
    """Compute recall@k — fraction of expected patterns found in top-k results.

    A retrieved source is considered relevant if any of the expected
    patterns appears as a substring (case-insensitive).
    """
    if not expected_patterns:
        return 1.0  # No expectations → trivially satisfied

    top_k = retrieved_sources[:k]
    found = 0
    for pattern in expected_patterns:
        pattern_lower = pattern.lower()
        if any(pattern_lower in src.lower() for src in top_k):
            found += 1
    return found / len(expected_patterns)


def precision_at_k(
    retrieved_sources: List[str],
    expected_patterns: List[str],
    k: int,
) -> float:
    """Compute precision@k — fraction of top-k results that are relevant."""
    if not expected_patterns:
        return 1.0

    top_k = retrieved_sources[:k]
    if not top_k:
        return 0.0

    relevant = 0
    for src in top_k:
        src_lower = src.lower()
        if any(pat.lower() in src_lower for pat in expected_patterns):
            relevant += 1
    return relevant / len(top_k)


def reciprocal_rank(
    retrieved_sources: List[str],
    expected_patterns: List[str],
) -> float:
    """Compute the reciprocal rank of the first relevant result."""
    if not expected_patterns:
        return 1.0

    for rank, src in enumerate(retrieved_sources, 1):
        src_lower = src.lower()
        if any(pat.lower() in src_lower for pat in expected_patterns):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def load_golden_dataset(path: str) -> List[Dict[str, Any]]:
    """Load the golden dataset from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def evaluate_retrieval(
    queries: List[Dict[str, Any]],
    top_k: int = 5,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run evaluation across all queries and compute aggregate metrics.

    Parameters
    ----------
    queries
        List of golden dataset entries.
    top_k
        Number of results to retrieve per query.
    dry_run
        If True, only evaluate intent classification (no Milvus needed).

    Returns
    -------
    dict
        Aggregate and per-query metrics.
    """
    # Lazy imports — only needed when actually running retrieval
    if not dry_run:
        from shared.rag_core import milvus_search
    from agent.router import classify

    per_query: List[Dict[str, Any]] = []
    total_recall = 0.0
    total_precision = 0.0
    total_rr = 0.0
    intent_correct = 0
    retrieval_queries = 0

    for entry in queries:
        query_id = entry["id"]
        query = entry["query"]
        expected_intent = entry["intent"]
        expected_sources = entry.get("expected_sources", [])

        # -- Intent classification --
        state = {"query": query}
        classified = classify(state)
        predicted_intent = classified["intent"]
        intent_match = predicted_intent == expected_intent
        if intent_match:
            intent_correct += 1

        result_entry: Dict[str, Any] = {
            "id": query_id,
            "query": query,
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "intent_correct": intent_match,
        }

        # -- Retrieval evaluation (skip for general/dry-run) --
        if not dry_run and expected_intent != "general":
            retrieval_queries += 1
            search_result = milvus_search(query, top_k=top_k)
            hits = search_result.get("results", [])

            # Extract source identifiers (file_path + citation_url)
            retrieved_sources = []
            for hit in hits:
                fp = hit.get("file_path", "")
                url = hit.get("citation_url", "")
                retrieved_sources.append(f"{fp} {url}")

            r_at_k = recall_at_k(retrieved_sources, expected_sources, top_k)
            p_at_k = precision_at_k(retrieved_sources, expected_sources, top_k)
            rr = reciprocal_rank(retrieved_sources, expected_sources)

            total_recall += r_at_k
            total_precision += p_at_k
            total_rr += rr

            result_entry.update(
                {
                    "recall_at_k": round(r_at_k, 4),
                    "precision_at_k": round(p_at_k, 4),
                    "reciprocal_rank": round(rr, 4),
                    "num_hits": len(hits),
                    "top_sources": [h.get("file_path", "") for h in hits[:3]],
                }
            )

        per_query.append(result_entry)

    # -- Aggregate metrics --
    n = len(queries)
    aggregate: Dict[str, Any] = {
        "total_queries": n,
        "intent_accuracy": round(intent_correct / n, 4) if n else 0,
    }

    if retrieval_queries > 0:
        aggregate.update(
            {
                "retrieval_queries": retrieval_queries,
                "mean_recall_at_k": round(total_recall / retrieval_queries, 4),
                "mean_precision_at_k": round(total_precision / retrieval_queries, 4),
                "mrr": round(total_rr / retrieval_queries, 4),
            }
        )

    return {"aggregate": aggregate, "per_query": per_query}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate docs-agent retrieval quality against a golden dataset."
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "golden_dataset.json"),
        help="Path to the golden dataset JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only evaluate intent classification (no Milvus needed).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write full results to this JSON file.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading golden dataset from %s", args.dataset)
    queries = load_golden_dataset(args.dataset)
    logger.info("Loaded %d queries", len(queries))

    if args.dry_run:
        logger.info("Dry-run mode: only evaluating intent classification")

    results = evaluate_retrieval(queries, top_k=args.top_k, dry_run=args.dry_run)

    # Print summary
    agg = results["aggregate"]
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total queries:        {agg['total_queries']}")
    print(f"  Intent accuracy:      {agg['intent_accuracy']:.1%}")

    if "mrr" in agg:
        print(f"  Retrieval queries:    {agg['retrieval_queries']}")
        print(f"  Mean Recall@{args.top_k}:       {agg['mean_recall_at_k']:.4f}")
        print(f"  Mean Precision@{args.top_k}:    {agg['mean_precision_at_k']:.4f}")
        print(f"  MRR:                  {agg['mrr']:.4f}")

    print("=" * 60)

    # Show per-query intent mismatches
    mismatches = [q for q in results["per_query"] if not q["intent_correct"]]
    if mismatches:
        print(f"\nIntent mismatches ({len(mismatches)}):")
        for m in mismatches:
            print(
                f"  [{m['id']}] {m['query']!r} "
                f"— expected={m['expected_intent']}, got={m['predicted_intent']}"
            )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
