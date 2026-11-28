"""HTTP client for the in-cluster TEI reranker service."""

from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

# Total budget for one rerank_hits call, shared by all of its batches.
DEFAULT_TIMEOUT_SEC = int(os.getenv("RERANKER_TIMEOUT_SEC", "30"))
# Keep at or below the reranker's --max-client-batch-size: TEI answers 413 above it.
MAX_BATCH = max(1, int(os.getenv("RERANKER_MAX_BATCH", "32")))


def _score_texts(service_url: str, query: str, texts: list[str], timeout: float) -> list[dict]:
    """Score every text in batches. Indices in the result refer to the full list."""
    deadline = time.monotonic() + timeout
    scores: list[dict] = []
    for start in range(0, len(texts), MAX_BATCH):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{timeout}s rerank budget used up after {start} of {len(texts)} texts")
        batch = texts[start : start + MAX_BATCH]
        response = requests.post(
            service_url,
            json={"query": query, "texts": batch},
            headers={"Content-Type": "application/json"},
            timeout=remaining,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"unexpected response format: {type(payload).__name__}")
        for item in payload:
            idx = item.get("index")
            # Check against this batch: an offset index could otherwise land on another batch's hit.
            if isinstance(idx, int) and 0 <= idx < len(batch):
                scores.append({**item, "index": start + idx})
    return scores


def enabled() -> bool:
    """True when a reranker is configured; callers then fetch extra candidates for it."""
    return bool((os.getenv("RERANKER_URL") or "").strip())


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
        payload = _score_texts(service_url, query, texts, timeout)
    except Exception as exc:
        # All or nothing: cross-encoder scores and Milvus distances are not comparable.
        logger.warning("Reranker failed, returning %d hits in original order: %s", len(hits), exc)
        return hits

    # TEI /rerank returns a list of {"index": int, "score": float}
    # sorted by score within each batch.
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
