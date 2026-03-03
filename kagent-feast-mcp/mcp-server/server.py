"""Kubeflow Docs MCP Server — semantic search over Kubeflow documentation.

Exposes a single ``search_kubeflow_docs`` tool via the MCP protocol so
that a Kagent Agent can retrieve relevant documentation chunks from a
Milvus vector store.
"""

from __future__ import annotations

import logging
import os
import threading

from fastmcp import FastMCP
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — read from environment with sensible defaults
# ---------------------------------------------------------------------------

MILVUS_URI: str = os.environ.get(
    "MILVUS_URI",
    "http://milvus.kubeflow.svc.cluster.local:19530",
)
MILVUS_USER: str = os.environ.get("MILVUS_USER", "root")
MILVUS_PASSWORD: str = os.environ.get("MILVUS_PASSWORD", "Milvus")
COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", "kubeflow_docs_docs_rag")
EMBEDDING_MODEL: str = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
)
PORT: int = int(os.environ.get("PORT", "8000"))

mcp = FastMCP("Kubeflow Docs MCP Server")

# ---------------------------------------------------------------------------
# Thread-safe lazy initialisation
# ---------------------------------------------------------------------------

_init_lock = threading.Lock()
model: SentenceTransformer | None = None
client: MilvusClient | None = None


def _init() -> None:
    """Initialise the embedding model and Milvus client once.

    Uses a lock so that concurrent requests do not race to create
    duplicate instances.
    """
    global model, client
    with _init_lock:
        if model is None:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            model = SentenceTransformer(EMBEDDING_MODEL)
        if client is None:
            logger.info("Connecting to Milvus at %s", MILVUS_URI)
            client = MilvusClient(
                uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD
            )


@mcp.tool()
def search_kubeflow_docs(query: str, top_k: int = 5) -> str:
    """Search Kubeflow documentation using semantic similarity.

    Args:
        query: The search query about Kubeflow.
        top_k: Number of results to return (default 5).

    Returns:
        Formatted search results with content and citation URLs.
    """
    try:
        _init()
    except Exception:
        logger.exception("Failed to initialise model or Milvus client")
        return "Error: unable to connect to the search backend. Please try again later."

    try:
        embedding = model.encode(query).tolist()  # type: ignore[union-attr]

        hits = client.search(  # type: ignore[union-attr]
            collection_name=COLLECTION_NAME,
            data=[embedding],
            limit=top_k,
            output_fields=["content_text", "citation_url", "file_path"],
        )[0]
    except Exception:
        logger.exception("Search failed for query: %s", query)
        return "Error: search request failed. Please try again later."

    if not hits:
        return "No results found for your query."

    results = []
    for i, hit in enumerate(hits, 1):
        entity = hit["entity"]
        entry = f"### Result {i} (score: {hit['distance']:.4f})"
        entry += f"\n**Source:** {entity.get('citation_url', '')}"
        entry += f"\n**File:** {entity.get('file_path', '')}"
        entry += f"\n\n{entity.get('content_text', '')}\n"
        results.append(entry)

    return "\n---\n".join(results)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORT)
