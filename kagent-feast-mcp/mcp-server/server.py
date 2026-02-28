import os
import requests
from fastmcp import FastMCP
from pymilvus import MilvusClient

MILVUS_URI = os.getenv("MILVUS_URI", "http://milvus.kubeflow.svc.cluster.local:19530")
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "Milvus")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "kubeflow_docs_docs_rag")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service.docs-agent.svc.cluster.local:8080")

mcp = FastMCP("Kubeflow Docs MCP Server")

client: MilvusClient = None


def _init():
    global client
    if client is None:
        client = MilvusClient(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)


def get_query_embedding(query: str) -> list:
    """Call the centralised embedding service (ADR-004)."""
    resp = requests.post(
        f"{EMBEDDING_SERVICE_URL}/embed",
        json={"texts": [query]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


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

    embedding = get_query_embedding(query)

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
    mcp.run(transport="stdio")
