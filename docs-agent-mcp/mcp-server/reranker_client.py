"""HTTP client for the in-cluster TEI reranker service."""

from __future__ import annotations

import os
import time

import requests

# Legacy single-timeout env override (if explicitly set by operator).
DEFAULT_TIMEOUT_SEC = int(os.getenv("RERANKER_TIMEOUT_SEC", "30"))
CONNECT_TIMEOUT_SEC = float(os.getenv("RERANKER_CONNECT_TIMEOUT_SEC", "1.0"))
BASE_READ_TIMEOUT_SEC = float(os.getenv("RERANKER_BASE_TIMEOUT_SEC", "1.5"))
PER_DOC_TIMEOUT_SEC = float(os.getenv("RERANKER_PER_DOC_TIMEOUT_SEC", "0.15"))
MAX_READ_TIMEOUT_SEC = float(os.getenv("RERANKER_MAX_TIMEOUT_SEC", "6.0"))
# Truncate candidate text sent to the CPU cross-encoder (~300 tokens covers title,
# section header, and lead paragraphs while avoiding O(L^2) attention blowup).
MAX_TEXT_CHARS = int(os.getenv("RERANKER_MAX_CHARS", "1200"))
COOLDOWN_SEC = float(os.getenv("RERANKER_COOLDOWN_SEC", "30.0"))

_cooldown_until_by_url: dict[str, float] = {}


def reset_circuit_breaker() -> None:
    """Clear any active cooldown state (used by tests or manual resets)."""
    _cooldown_until_by_url.clear()


def _is_cooling_down(service_url: str) -> bool:
    until = _cooldown_until_by_url.get(service_url, 0.0)
    if until <= 0.0:
        return False
    if time.monotonic() < until:
        return True
    _cooldown_until_by_url.pop(service_url, None)
    return False


def _scaled_timeout(num_hits: int, total_chars: int) -> tuple[float, float]:
    """Scale read timeout by candidate count and character volume, capped at MAX_READ_TIMEOUT_SEC."""
    connect_sec = float(os.getenv("RERANKER_CONNECT_TIMEOUT_SEC", str(CONNECT_TIMEOUT_SEC)))
    base_sec = float(os.getenv("RERANKER_BASE_TIMEOUT_SEC", str(BASE_READ_TIMEOUT_SEC)))
    per_doc_sec = float(os.getenv("RERANKER_PER_DOC_TIMEOUT_SEC", str(PER_DOC_TIMEOUT_SEC)))
    max_read_sec = float(os.getenv("RERANKER_MAX_TIMEOUT_SEC", str(MAX_READ_TIMEOUT_SEC)))
    read_sec = base_sec + (per_doc_sec * max(1, num_hits)) + (max(0, total_chars) / 20000.0)
    return (connect_sec, round(min(read_sec, max_read_sec), 3))


def enabled() -> bool:
    """True when a reranker is configured and not in a failure cooldown window."""
    service_url = (os.getenv("RERANKER_URL") or "").strip()
    if not service_url:
        return False
    return not _is_cooling_down(service_url)


def rerank_hits(
    query: str,
    hits: list[dict],
    timeout: int | float | tuple[float, float] | None = None,
) -> list[dict]:
    """Score Milvus hits using the cross-encoder and reorder them."""
    if not hits:
        return []

    service_url = (os.getenv("RERANKER_URL") or "").strip()
    if not service_url:
        # If no reranker is configured, just return hits as-is.
        return hits

    if _is_cooling_down(service_url):
        print("Reranker skipped: circuit breaker cooldown active")
        return hits

    max_chars = int(os.getenv("RERANKER_MAX_CHARS", str(MAX_TEXT_CHARS)))
    texts = []
    for hit in hits:
        raw_text = str(hit.get("entity", {}).get("content_text", ""))
        texts.append(raw_text[:max_chars] if max_chars > 0 else raw_text)

    if timeout is None:
        explicit_env = os.getenv("RERANKER_TIMEOUT_SEC")
        if explicit_env:
            effective_timeout: int | float | tuple[float, float] = float(explicit_env)
        else:
            total_chars = sum(len(t) for t in texts)
            effective_timeout = _scaled_timeout(len(texts), total_chars)
    else:
        effective_timeout = timeout

    try:
        response = requests.post(
            service_url,
            json={"query": query, "texts": texts},
            headers={"Content-Type": "application/json"},
            timeout=effective_timeout,
        )
        response.raise_for_status()
    except Exception as exc:
        cooldown_sec = float(os.getenv("RERANKER_COOLDOWN_SEC", str(COOLDOWN_SEC)))
        if cooldown_sec > 0:
            _cooldown_until_by_url[service_url] = time.monotonic() + cooldown_sec
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
