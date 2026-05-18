import os
import re
import threading

from fastmcp import FastMCP
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "Milvus")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs_rag")
ISSUES_COLLECTION_NAME = os.getenv("ISSUES_COLLECTION_NAME", "issues_rag")
CODE_COLLECTION_NAME = os.getenv("CODE_COLLECTION_NAME", "code_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
PORT = int(os.getenv("PORT", "8000"))

mcp = FastMCP("Kubeflow Docs MCP Server")

model: SentenceTransformer = None
client: MilvusClient = None
_init_lock = threading.Lock()

_FILTER_VALUE_RE = re.compile(r"^[A-Za-z0-9_/.\-]+$")


def _init():
    global model, client
    if model is not None and client is not None:
        return
    with _init_lock:
        if model is None:
            model = SentenceTransformer(EMBEDDING_MODEL)
        if client is None:
            client = MilvusClient(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)


def _search_collection(
    collection_name: str, query: str, top_k: int, output_fields: list[str], filter_expr: str = ""
) -> list[dict]:
    """Shared helper: encode query and search a Milvus collection."""
    _init()
    # Search requires the collection to be loaded; pipelines may release load after ingest.
    try:
        client.load_collection(collection_name)
    except Exception as e:
        # Missing collection or load failure — surface on search
        raise RuntimeError(f"Milvus load_collection failed for {collection_name}: {e}") from e
    embedding = model.encode(query).tolist()
    search_params = {
        "collection_name": collection_name,
        "data": [embedding],
        "limit": top_k,
        "output_fields": output_fields,
    }
    if filter_expr:
        search_params["filter"] = filter_expr
    hits = client.search(**search_params)[0]
    return hits


def _safe_filter_value(name: str, value: str) -> str:
    """Validate user-controlled values before interpolating Milvus expressions."""
    if not _FILTER_VALUE_RE.fullmatch(value):
        raise ValueError(f"Invalid {name} filter value: {value!r}")
    return value


@mcp.tool()
def search_kubeflow_docs(query: str, top_k: int = 5) -> str:
    """Search Kubeflow documentation using semantic similarity.

    Args:
        query: The search query about Kubeflow.
        top_k: Number of results to return (default 5).

    Returns:
        Formatted search results with content and citation URLs.
    """
    hits = _search_collection(
        COLLECTION_NAME,
        query,
        top_k,
        ["content_text", "citation_url", "file_path"],
    )

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


@mcp.tool()
def search_github_issues(query: str, top_k: int = 5, repo: str = "", state: str = "") -> str:
    """Search Kubeflow GitHub issues for bug reports, troubleshooting,
    and community solutions. Searches across all repositories including
    kubeflow/kubeflow, kubeflow/pipelines, kubeflow/manifests,
    kubeflow/katib, kserve/kserve, and kubeflow/website.

    Args:
        query: Search query about errors, bugs, or issues.
        top_k: Number of results to return (default 5).
        repo: Optional filter by repository (e.g., "kubeflow/kubeflow").
        state: Optional filter by issue state ("open" or "closed").

    Returns:
        Formatted search results with content and GitHub issue URLs.
    """
    filters = []
    if repo:
        repo = _safe_filter_value("repo", repo)
        filters.append(f'repo_name == "{repo}"')
    if state:
        state = _safe_filter_value("state", state)
        filters.append(f'issue_state == "{state}"')
    filter_expr = " and ".join(filters)

    hits = _search_collection(
        ISSUES_COLLECTION_NAME,
        query,
        top_k,
        ["content_text", "citation_url", "repo_name", "issue_number", "issue_state", "issue_labels"],
        filter_expr=filter_expr,
    )

    if not hits:
        return "No issues found for your query."

    results = []
    for i, hit in enumerate(hits, 1):
        entity = hit["entity"]
        entry = f"### Result {i} (score: {hit['distance']:.4f})"
        entry += f"\n**Source:** {entity.get('citation_url', '')}"
        entry += f"\n**Repo:** {entity.get('repo_name', '')}"

        issue_num = entity.get("issue_number", "")
        state = entity.get("issue_state", "")
        labels = entity.get("issue_labels", "")
        if issue_num:
            entry += f"\n**Issue:** #{issue_num}"
        if state:
            entry += f" ({state})"
        if labels:
            entry += f"\n**Labels:** {labels}"

        entry += f"\n\n{entity.get('content_text', '')}\n"
        results.append(entry)

    return "\n---\n".join(results)


@mcp.tool()
def search_kubeflow_code(query: str, top_k: int = 5, resource_kind: str = "") -> str:
    """Search Kubeflow code and YAML manifests using semantic similarity.

    Use this tool when the user asks about Kubernetes manifests, YAML
    configurations, Deployments, Services, ConfigMaps, Python source code,
    or infrastructure definitions in the Kubeflow project.

    Args:
        query: The search query about Kubeflow code or manifests.
        top_k: Number of results to return (default 5).
        resource_kind: Optional filter by Kubernetes resource kind
            (e.g., "Deployment", "Service", "ConfigMap", "StatefulSet",
            "ServiceAccount", "ClusterRole", "Role", "RoleBinding").
            When set, only chunks matching this exact kind are returned.
            For Python code use "function", "class", or "module".
            Leave empty to search all resource types.

    Returns:
        Formatted search results with code content, resource metadata,
        and citation URLs.
    """
    if resource_kind:
        resource_kind = _safe_filter_value("resource_kind", resource_kind)
    filter_expr = f"resource_kind == '{resource_kind}'" if resource_kind else ""
    hits = _search_collection(
        CODE_COLLECTION_NAME,
        query,
        top_k,
        [
            "content_text",
            "citation_url",
            "file_path",
            "resource_kind",
            "resource_name",
            "resource_namespace",
            "file_type",
        ],
        filter_expr=filter_expr,
    )

    if not hits:
        return "No code results found for your query."

    results = []
    for i, hit in enumerate(hits, 1):
        entity = hit["entity"]
        entry = f"### Result {i} (score: {hit['distance']:.4f})"
        entry += f"\n**Source:** {entity.get('citation_url', '')}"
        entry += f"\n**File:** {entity.get('file_path', '')}"

        kind = entity.get("resource_kind", "")
        name = entity.get("resource_name", "")
        ns = entity.get("resource_namespace", "")
        ftype = entity.get("file_type", "")
        if kind or name:
            entry += f"\n**Resource:** {kind}"
            if name:
                entry += f" `{name}`"
            if ns:
                entry += f" (namespace: {ns})"
        if ftype:
            entry += f"\n**Type:** {ftype}"

        entry += f"\n\n```\n{entity.get('content_text', '')}\n```\n"
        results.append(entry)

    return "\n---\n".join(results)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORT)
