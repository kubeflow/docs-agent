import os

from fastmcp import FastMCP
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "Milvus")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kubeflow_docs_docs_rag")
ISSUES_COLLECTION_NAME = os.getenv("ISSUES_COLLECTION_NAME", "issues_rag")
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


def _search_collection(
    collection_name: str,
    query: str,
    top_k: int,
    output_fields: list,
    filter_expr: str = "",
) -> str:
    """Shared search-and-format logic for all MCP search tools."""
    _init()

    embedding = model.encode(query).tolist()

    search_params = dict(
        collection_name=collection_name,
        data=[embedding],
        limit=top_k,
        output_fields=output_fields,
    )
    if filter_expr:
        search_params["filter"] = filter_expr

    hits = client.search(**search_params)[0]

    if not hits:
        return "No results found for your query."

    results = []
    for i, hit in enumerate(hits, 1):
        entity = hit["entity"]
        entry = f"### Result {i} (score: {hit['distance']:.4f})"
        entry += f"\n**Source:** {entity.get('citation_url', '')}"
        # Render additional fields (skip content_text and citation_url)
        for field in output_fields:
            if field in ("content_text", "citation_url"):
                continue
            val = entity.get(field, "")
            if val != "" and val is not None:
                entry += f"\n**{field}:** {val}"
        entry += f"\n\n{entity.get('content_text', '')}\n"
        results.append(entry)

    return "\n---\n".join(results)


@mcp.tool()
def search_kubeflow_docs(query: str, top_k: int = 5) -> str:
    """Search Kubeflow documentation using semantic similarity.

    Args:
        query: The search query about Kubeflow.
        top_k: Number of results to return (default 5).

    Returns:
        Formatted search results with content and citation URLs.
    """
    return _search_collection(
        collection_name=COLLECTION_NAME,
        query=query,
        top_k=top_k,
        output_fields=["content_text", "citation_url", "file_path"],
    )


@mcp.tool()
def search_github_issues(
    query: str,
    top_k: int = 5,
) -> str:
    """Search Kubeflow GitHub issues for bug reports, troubleshooting,
    and community solutions. Searches across all repositories including
    kubeflow/kubeflow, kubeflow/pipelines, kubeflow/manifests,
    kubeflow/katib, kserve/kserve, and kubeflow/website.

    Args:
        query: Search query about errors, bugs, or issues.
        top_k: Number of results to return (default 5).

    Returns:
        Formatted search results with content and GitHub issue URLs.
    """
    return _search_collection(
        collection_name=ISSUES_COLLECTION_NAME,
        query=query,
        top_k=top_k,
        output_fields=[
            "content_text", "citation_url", "repo_name",
            "issue_number", "issue_state", "issue_labels",
        ],
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORT)
