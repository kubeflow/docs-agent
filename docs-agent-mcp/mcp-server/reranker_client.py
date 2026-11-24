"""HTTP client for the in-cluster TEI reranker service."""

from __future__ import annotations

import math
import os

import requests

DEFAULT_TIMEOUT_SEC = int(os.getenv("RERANKER_TIMEOUT_SEC", "30"))


def _is_valid_index(value: object, hit_count: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < hit_count


def _parse_valid_score(item: dict) -> float | None:
    if "score" not in item:
        return None
    score = item["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    try:
        score_f = float(score)
    except OverflowError:
        return None
    if not math.isfinite(score_f):
        return None
    return score_f


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
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"Reranker failed: {exc}")
        return hits

    if not isinstance(payload, list):
        print(f"Reranker returned unexpected format: {type(payload)}")
        return hits

    # TEI /rerank returns a list of {"index": int, "score": float}.
    # Validate entries strictly so malformed items cannot crash or duplicate hits.
    scored: list[tuple[float, int, dict]] = []
    seen_indices: set[int] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not _is_valid_index(idx, len(hits)) or idx in seen_indices:
            continue
        score = _parse_valid_score(item)
        if score is None:
            continue

        seen_indices.add(idx)
        hit = hits[idx].copy()
        # Replace distance with cross-encoder score so downstream can see it
        hit["distance"] = score
        hit["reranked"] = True
        scored.append((score, idx, hit))

    if not scored:
        print("Reranker returned no valid scores; keeping original order")
        return hits

    # Sort only cross-encoder-scored hits (score descending, original index ascending),
    # then append any unreranked hits in their original relative order without mixing scales.
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    reranked = [entry[2] for entry in scored]
    unreranked = [hit for i, hit in enumerate(hits) if i not in seen_indices]

    if unreranked:
        print(
            f"Reranker: scored and reordered {len(reranked)} hits via cross-encoder "
            f"({len(unreranked)} unreranked fallback)"
        )
    else:
        print(f"Reranker: scored and reordered {len(reranked)} hits via cross-encoder")
    return reranked + unreranked
