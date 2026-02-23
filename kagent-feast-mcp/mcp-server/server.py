import os
import logging

from fastmcp import FastMCP
from feast import FeatureStore
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "/app/feast_repo")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
PORT = int(os.getenv("PORT", "8000"))

mcp = FastMCP(
    "Kubeflow Docs MCP Server",
    description="Search Kubeflow documentation via Feast vector store",
)

store: FeatureStore = None
model: SentenceTransformer = None


def _init():
    global store, model
    if store is None:
        logger.info("Loading Feast store from %s", FEAST_REPO_PATH)
        store = FeatureStore(repo_path=FEAST_REPO_PATH)
    if model is None:
        logger.info("Loading embedding model %s", EMBEDDING_MODEL)
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Model loaded")


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

    response = store.retrieve_online_documents_v2(
        feature="docs_rag:vector",
        query=embedding,
        top_k=top_k,
    )

    if response is None or response.empty:
        return "No results found for your query."

    results = []
    for i, row in response.iterrows():
        content = row.get("content_text", "")
        url = row.get("citation_url", "")
        file_path = row.get("file_path", "")
        score = row.get("distance", "")

        entry = f"### Result {i + 1}"
        if score:
            entry += f" (score: {score:.4f})"
        entry += f"\n**Source:** {url}\n**File:** {file_path}\n\n{content}\n"
        results.append(entry)

    return "\n---\n".join(results)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORT)