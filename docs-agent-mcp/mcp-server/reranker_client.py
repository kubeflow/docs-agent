"""HTTP client for the in-cluster TEI reranker service."""

from __future__ import annotations

import os

import requests

DEFAULT_TIMEOUT_SEC = int(os.getenv("RERANKER_TIMEOUT_SEC", "30"))


def rerank_hits(query: str, hits: list[dict], timeout: int = DEFAULT_TIMEOUT_SEC) -> list[dict]:
    """Score Milvus hits using the cross-encoder and reorder them."""
    if not hits:
        return []

    service_url = (os.getenv("RERANKER_URL") or "").strip()
    if not service_url:
        # If no reranker is configured, just return hits as-is.
        return hits

    # Extract text to rerank
    texts = [str(hit.get("entity", {}).get("content_text", "")) for hit in hits]

    try:
        response = requests.post(
            service_url,
            json={"query": query, "texts": texts},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception as exc:
        print(f"Reranker failed: {exc}")
        return hits

    payload = response.json()
    if not isinstance(payload, list):
        print(f"Reranker returned unexpected format: {type(payload)}")
        return hits

    # TEI /rerank returns a list of {"index": int, "score": float}
    # It returns them sorted by score.
    reranked = []
    for item in payload:
        idx = item.get("index")
        score = item.get("score", 0.0)
        if isinstance(idx, int) and 0 <= idx < len(hits):
            hit = hits[idx].copy()
            # Replace distance with cross-encoder score so downstream can see it
            hit["distance"] = score
            hit["reranked"] = True
            reranked.append(hit)

    # Fallback for hits that weren't returned
    seen_indices = {item.get("index") for item in payload if isinstance(item.get("index"), int)}
    for i, hit in enumerate(hits):
        if i not in seen_indices:
            reranked.append(hit)

    # Sort strictly by the new cross-encoder score descending
    reranked.sort(key=lambda x: x.get("distance", -float("inf")), reverse=True)
    print(f"Reranker: scored and reordered {len(reranked)} hits via cross-encoder")
    return reranked
