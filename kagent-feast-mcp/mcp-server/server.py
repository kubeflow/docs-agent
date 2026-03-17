import os
from fastmcp import FastMCP
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# ============================================================================
# Configuration - Override via environment variables (with production defaults)
# ============================================================================
# Verified working as of March 2026
#
# Key namespaces:
# - Default namespace: docs-agent (must match setup.yaml and MCP URL)
# - Collection name: matches Kubeflow pipeline output (kubeflow_docs_docs_rag)
#
# For production, set these via K8s manifests:
#   MILVUS_URI, MILVUS_USER, MILVUS_PASSWORD, COLLECTION_NAME, EMBEDDING_MODEL
# ============================================================================

MILVUS_URI = os.getenv("MILVUS_URI", "http://milvus.docs-agent.svc.cluster.local:19530")
MILVUS_USER = os.getenv("MILVUS_USER", "root")  # Change in production!
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "Milvus")  # Change in production!
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kubeflow_docs_docs_rag")
# Default model: all-mpnet-base-v2 (768-dim, high quality)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
PORT = int(os.getenv("PORT", "8000"))

mcp = FastMCP("Kubeflow Docs MCP Server")

model: SentenceTransformer = None
client: MilvusClient = None


def _init():
    global model, client
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)
    if client is None:
        client = MilvusClient(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)


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

    hits = client.search(
        collection_name=COLLECTION_NAME,
        data=[embedding],
        limit=top_k,
        output_fields=["content_text", "citation_url", "file_path"],
    )[0]

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
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORT)
