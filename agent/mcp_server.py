"""Kagent-native MCP server for Kubeflow docs-agent.

Exposes partition-aware retrieval tools that the Kagent Agent CRD
orchestrates via its system prompt.  Each tool wraps a quality-aware
search pipeline::

    encode query  →  search Milvus partition  →  evaluate avg similarity
        →  (if below threshold) broaden query & retry  →  format results

The Agent CRD's ``systemMessage`` decides *which* tool to call based on
the user's question; the tool decides *how* to search and whether the
results are good enough to return.

Usage (local development)::

    python -m agent.mcp_server          # streamable-http on :8000

Usage (Kubernetes)::

    # See agent/manifests/ for Agent CRD, ModelConfig, and MCP deployment.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure the project root is importable so ``shared.*`` resolves.
# ---------------------------------------------------------------------------
try:
    from shared.rag_core import (
        EMBEDDING_MODEL,
        MILVUS_COLLECTION,
        MILVUS_HOST,
        MILVUS_PORT,
        MILVUS_VECTOR_FIELD,
    )
except ModuleNotFoundError:
    _project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from shared.rag_core import (  # noqa: E402
        EMBEDDING_MODEL,
        MILVUS_COLLECTION,
        MILVUS_HOST,
        MILVUS_PORT,
        MILVUS_VECTOR_FIELD,
    )

from fastmcp import FastMCP
from pymilvus import Collection, connections, utility
from sentence_transformers import SentenceTransformer

from agent.config import (
    DEFAULT_TOP_K,
    MAX_RETRIEVAL_ATTEMPTS,
    PARTITION_MAP,
    RELEVANCE_THRESHOLD,
    Intent,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PORT = int(os.getenv("MCP_PORT", os.getenv("PORT", "8000")))

# ---------------------------------------------------------------------------
# Thread-safe singleton: embedding encoder
# ---------------------------------------------------------------------------

_encoder: Optional[SentenceTransformer] = None
_encoder_lock = threading.Lock()


def _get_encoder() -> SentenceTransformer:
    """Lazily load the SentenceTransformer model (thread-safe).

    Uses a double-checked locking pattern so that concurrent requests
    do not each trigger an expensive model load.
    """
    global _encoder
    if _encoder is not None:
        return _encoder
    with _encoder_lock:
        if _encoder is None:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            _encoder = SentenceTransformer(EMBEDDING_MODEL)
        return _encoder  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Thread-safe singleton: Milvus connection
# ---------------------------------------------------------------------------

_milvus_connected = False
_milvus_lock = threading.Lock()


def _ensure_milvus() -> None:
    """Guarantee a healthy Milvus connection, reconnecting if stale.

    This is safe to call on every request — it short-circuits when the
    existing connection is still alive and only reconnects when the
    link has been lost.
    """
    global _milvus_connected
    if _milvus_connected:
        try:
            utility.list_collections(using="default")
            return
        except Exception:
            _milvus_connected = False

    with _milvus_lock:
        # Re-check after acquiring lock (another thread may have reconnected)
        if _milvus_connected:
            try:
                utility.list_collections(using="default")
                return
            except Exception:
                _milvus_connected = False

        try:
            connections.disconnect(alias="default")
        except Exception:
            pass
        logger.info(
            "Connecting to Milvus at %s:%s", MILVUS_HOST, MILVUS_PORT
        )
        connections.connect(
            alias="default", host=MILVUS_HOST, port=MILVUS_PORT
        )
        _milvus_connected = True


# ---------------------------------------------------------------------------
# Core search implementation with partition & quality awareness
# ---------------------------------------------------------------------------


def _search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    partition: str = "",
    auto_broaden: bool = True,
) -> List[Dict[str, Any]]:
    """Partition-aware semantic search with quality-based broadening.

    Parameters
    ----------
    query
        The user's search query.
    top_k
        Maximum number of results to return.
    partition
        Milvus partition name.  Empty string searches the full collection.
    auto_broaden
        If ``True`` and the first search yields low-quality results
        (avg similarity below ``RELEVANCE_THRESHOLD``), broaden the
        query and retry up to ``MAX_RETRIEVAL_ATTEMPTS`` times.

    Returns
    -------
    list[dict]
        Hit dicts with keys: ``similarity``, ``file_path``,
        ``citation_url``, ``content_text``.
    """
    _ensure_milvus()
    encoder = _get_encoder()
    collection = Collection(MILVUS_COLLECTION, using="default")
    collection.load()

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}
    partition_names = [partition] if partition else None

    best_hits: List[Dict[str, Any]] = []

    for attempt in range(1, MAX_RETRIEVAL_ATTEMPTS + 1):
        current_query = query
        if attempt > 1 and auto_broaden:
            current_query = f"{query} overview setup getting started"
            logger.info(
                "Broadened query (attempt %d/%d): %r",
                attempt,
                MAX_RETRIEVAL_ATTEMPTS,
                current_query,
            )

        query_vec = encoder.encode(current_query).tolist()

        search_kwargs: Dict[str, Any] = {
            "data": [query_vec],
            "anns_field": MILVUS_VECTOR_FIELD,
            "param": search_params,
            "limit": int(top_k),
            "output_fields": ["file_path", "content_text", "citation_url"],
        }
        if partition_names:
            search_kwargs["partition_names"] = partition_names

        results = collection.search(**search_kwargs)

        hits: List[Dict[str, Any]] = []
        for hit in results[0]:
            similarity = 1.0 - float(hit.distance)
            entity = hit.entity
            content_text = entity.get("content_text") or ""
            if isinstance(content_text, str) and len(content_text) > 500:
                content_text = content_text[:500] + "..."
            hits.append(
                {
                    "similarity": similarity,
                    "file_path": entity.get("file_path"),
                    "citation_url": entity.get("citation_url"),
                    "content_text": content_text,
                }
            )

        if not hits:
            if attempt < MAX_RETRIEVAL_ATTEMPTS and auto_broaden:
                continue
            return best_hits

        avg_sim = sum(h["similarity"] for h in hits) / len(hits)
        logger.info(
            "Search attempt %d: %d hits, avg_similarity=%.3f "
            "(threshold=%.3f)",
            attempt,
            len(hits),
            avg_sim,
            RELEVANCE_THRESHOLD,
        )

        # Good enough — return immediately
        if avg_sim >= RELEVANCE_THRESHOLD or not auto_broaden:
            return hits

        # Keep the best results seen so far
        if not best_hits or avg_sim > (
            sum(h["similarity"] for h in best_hits) / len(best_hits)
        ):
            best_hits = hits

    return best_hits


def _format_results(hits: List[Dict[str, Any]]) -> str:
    """Format search hits into a Markdown string for the LLM."""
    if not hits:
        return "No relevant results found in the documentation."

    parts: List[str] = []
    for i, hit in enumerate(hits, 1):
        entry = f"### Result {i} (similarity: {hit['similarity']:.4f})"
        if hit.get("citation_url"):
            entry += f"\n**Source:** {hit['citation_url']}"
        if hit.get("file_path"):
            entry += f"\n**File:** {hit['file_path']}"
        entry += f"\n\n{hit.get('content_text', '')}\n"
        parts.append(entry)

    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "Kubeflow Docs Agent",
    description=(
        "Partition-aware retrieval tools for the Kubeflow documentation "
        "assistant.  Use these tools to search official Kubeflow docs "
        "and platform architecture documentation backed by Milvus."
    ),
)


@mcp.tool()
def search_kubeflow_docs(query: str, top_k: int = 5) -> str:
    """Search Kubeflow documentation using semantic similarity.

    Use this tool when the user asks about Kubeflow-specific topics:
    Pipelines, KServe, Katib, Notebooks/Jupyter, SDK, CLI, APIs,
    training operators (TFJob, PyTorchJob, MPIJob, XGBoostJob),
    installation, configuration, errors, or release details.

    The tool automatically evaluates result quality and broadens the
    search query if the initial results are not relevant enough.

    Args:
        query: A clear, focused search query about Kubeflow.  Refine
            the user's question into a documentation-specific search
            string (e.g. "KServe InferenceService canary rollout").
        top_k: Number of results to return (1–10, default 5).

    Returns:
        Formatted search results with content excerpts and source URLs.
    """
    top_k = max(1, min(10, top_k))
    partition = PARTITION_MAP.get(Intent.KUBEFLOW_DOCS, "")
    logger.info(
        "search_kubeflow_docs: query=%r top_k=%d partition=%r",
        query,
        top_k,
        partition,
    )
    try:
        hits = _search(query, top_k=top_k, partition=partition)
        return _format_results(hits)
    except Exception:
        logger.exception("search_kubeflow_docs failed")
        return "Search failed due to an internal error. Please try again."


@mcp.tool()
def search_platform_docs(query: str, top_k: int = 5) -> str:
    """Search platform architecture and infrastructure documentation.

    Use this tool when the user asks about deployment, infrastructure,
    Kubernetes cluster setup, Terraform, OCI/OKE, Helm charts, Istio,
    Knative, cert-manager, Dex/OIDC, GPU node pools, autoscaling,
    Kustomize, ArgoCD, or GitOps workflows.

    The tool searches the ``platform_arch`` Milvus partition populated
    by the platform architecture ingestion pipeline.

    Args:
        query: A focused search query about platform/infrastructure
            topics (e.g. "Terraform OKE GPU node pool configuration").
        top_k: Number of results to return (1–10, default 5).

    Returns:
        Formatted search results with content excerpts and source URLs.
    """
    top_k = max(1, min(10, top_k))
    partition = PARTITION_MAP.get(Intent.PLATFORM_ARCH, "platform_arch")
    logger.info(
        "search_platform_docs: query=%r top_k=%d partition=%r",
        query,
        top_k,
        partition,
    )
    try:
        hits = _search(query, top_k=top_k, partition=partition)
        return _format_results(hits)
    except Exception:
        logger.exception("search_platform_docs failed")
        return "Search failed due to an internal error. Please try again."


@mcp.tool()
def search_all_docs(query: str, top_k: int = 5) -> str:
    """Search across all documentation partitions.

    Use this tool when the user's question spans both Kubeflow features
    AND infrastructure/deployment, or when you are unsure which
    partition is most relevant.

    Args:
        query: A search query that may span multiple documentation areas.
        top_k: Number of results to return (1–10, default 5).

    Returns:
        Formatted search results with content excerpts and source URLs.
    """
    top_k = max(1, min(10, top_k))
    logger.info("search_all_docs: query=%r top_k=%d", query, top_k)
    try:
        hits = _search(query, top_k=top_k, partition="")
        return _format_results(hits)
    except Exception:
        logger.exception("search_all_docs failed")
        return "Search failed due to an internal error. Please try again."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(
        "Starting Kubeflow Docs Agent MCP server on port %d", PORT
    )
    # Pre-load the embedding model at startup so the first request is fast
    _get_encoder()
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORT)
