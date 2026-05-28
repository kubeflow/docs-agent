from fastmcp import FastMCP
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

MILVUS_URI = "http://milvus.<YOUR_NAMESPACE>.svc.cluster.local:19530"
MILVUS_USER = "root"
MILVUS_PASSWORD = "Milvus"
COLLECTION_NAME = "kubeflow_docs_docs_rag"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
PORT = 8000

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
