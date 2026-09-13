"""HTTP client for the in-cluster TEI embeddings service."""

from __future__ import annotations

import os
from typing import Sequence

import requests

DEFAULT_EMBEDDINGS_URL = "http://embeddings-service-predictor.ml-infra.svc.cluster.local/embed"
DEFAULT_TIMEOUT_SEC = int(os.getenv("EMBEDDINGS_TIMEOUT_SEC", "60"))
# TEI all-mpnet-base-v2: each input must be <384 tokens.
MAX_TEI_INPUT_CHARS = int(os.getenv("MAX_TEI_INPUT_CHARS", "1000"))


def embed_texts(
    texts: Sequence[str],
    *,
    url: str | None = None,
    batch_size: int = 8,
    timeout: int = DEFAULT_TIMEOUT_SEC,
) -> list[list[float]]:
    """Return embedding vectors for each input string (768-dim for all-mpnet-base-v2)."""
    if not texts:
        return []

    service_url = (url or os.getenv("EMBEDDINGS_URL") or DEFAULT_EMBEDDINGS_URL).strip()
    if not service_url:
        raise ValueError("EMBEDDINGS_URL is not configured")

    batch_size = max(1, int(batch_size))
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = [t[:MAX_TEI_INPUT_CHARS] for t in texts[start : start + batch_size]]
        response = requests.post(
            service_url,
            json={"inputs": batch},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) != len(batch):
            raise RuntimeError(
                f"Embeddings service returned {len(payload) if isinstance(payload, list) else type(payload)} "
                f"vectors for batch of {len(batch)}"
            )
        vectors.extend(payload)

    return vectors


def embed_query(query: str, **kwargs) -> list[float]:
    """Embed a single search query."""
    return embed_texts([query], **kwargs)[0]
