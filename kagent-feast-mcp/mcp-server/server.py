from fastmcp import FastMCP
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from shared.reranking import candidate_pool_limit, load_rerank_config_from_env, rerank_documents

MILVUS_URI = os.getenv("MILVUS_URI", "http://milvus.<YOUR_NAMESPACE>.svc.cluster.local:19530")
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "Milvus")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kubeflow_docs_docs_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
PORT = int(os.getenv("PORT", "8000"))
RERANK_CONFIG = load_rerank_config_from_env()

mcp = FastMCP("Kubeflow Docs MCP Server")
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

model: SentenceTransformer = None
client: MilvusClient = None
_initialized = False
_init_lock = threading.Lock()


def _init():
    """Initialize shared model/client exactly once.

    Synchronization strategy:
    - Fast path: return immediately after successful initialization.
    - Slow path: take a process-local lock and re-check state (double-checked locking).

    This guarantees that concurrent callers block until initialization completes,
    and all callers observe the same initialized instances.
    """
    global model, client, _initialized

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        logger.info("Initializing shared MCP resources")

        # Build local instances first, then publish atomically under the lock.
        local_model = SentenceTransformer(EMBEDDING_MODEL)
        local_client = MilvusClient(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)

        model = local_model
        client = local_client
        _initialized = True

        logger.info("Shared MCP resources initialized")


@mcp.tool()
def search_kubeflow_docs(query: str, top_k: int = 5) -> str:
    """Search Kubeflow documentation using semantic similarity.

    Args:
        query: The search query about Kubeflow.
        top_k: Number of results to return (default 5).

    Returns:
        Formatted search results with content and citation URLs.
    """
    _init()

    embedding = model.encode(query).tolist()
    requested_top_k = max(1, int(top_k))
    candidate_limit = candidate_pool_limit(requested_top_k, RERANK_CONFIG)

    hits = client.search(
        collection_name=COLLECTION_NAME,
        data=[embedding],
        limit=candidate_limit,
        output_fields=["content_text", "citation_url", "file_path"],
    )[0]

    if not hits:
        return "No results found for your query."

    docs: List[Dict[str, Any]] = []
    for hit in hits:
        entity = hit.get("entity", {})
        content_text = entity.get("content_text") or ""
        if isinstance(content_text, str) and len(content_text) > 400:
            content_text = content_text[:400] + "..."

        docs.append(
            {
                "distance": float(hit.get("distance", 0.0)),
                "similarity": 1.0 - float(hit.get("distance", 0.0)),
                "file_path": entity.get("file_path"),
                "citation_url": entity.get("citation_url"),
                "content_text": content_text,
            }
        )

    selected_docs = rerank_documents(
        query=query,
        docs=docs,
        config=RERANK_CONFIG,
        top_k=requested_top_k,
        logger=logger,
        log_prefix="mcp_search",
    )

    results = []
    for i, doc in enumerate(selected_docs, 1):
        entry = f"### Result {i} (rerank_score: {doc.get('rerank_score', 0.0):.4f})"
        entry += f"\n**Similarity:** {doc.get('similarity', 0.0):.4f}"
        entry += f"\n**Keyword Score:** {doc.get('keyword_score', 0.0):.4f}"
        entry += f"\n**Metadata Score:** {doc.get('metadata_score', 0.0):.4f}"
        entry += f"\n**Source:** {doc.get('citation_url', '')}"
        entry += f"\n**File:** {doc.get('file_path', '')}"
        entry += f"\n\n{doc.get('content_text', '')}\n"
        results.append(entry)

    return "\n---\n".join(results)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORT)
