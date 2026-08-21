import json
import os
import re
import threading

from fastmcp import FastMCP
from pymilvus import MilvusClient

from rag_collections import CODE_COLLECTION, DOCS_COLLECTION, ISSUES_COLLECTION
from embeddings_client import embed_query

MILVUS_URI = os.getenv("MILVUS_URI", "http://milvus-milvus.ml-infra.svc.cluster.local:19530")
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", DOCS_COLLECTION)
ISSUES_COLLECTION_NAME = os.getenv("ISSUES_COLLECTION_NAME", ISSUES_COLLECTION)
CODE_COLLECTION_NAME = os.getenv("CODE_COLLECTION_NAME", CODE_COLLECTION)
EMBEDDINGS_URL = os.getenv("EMBEDDINGS_URL", "")
PORT = int(os.getenv("PORT", "8000"))

mcp = FastMCP("Kubeflow Docs MCP Server")

client: MilvusClient | None = None
_init_lock = threading.Lock()

_FILTER_VALUE_RE = re.compile(r"^[A-Za-z0-9_/.\-]+$")
_SEARCH_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
DOCS_CONTEXT_MAX_CHUNKS = 20
DOCS_CONTEXT_MAX_CHARS = 24_000


def _init():
    global client
    if client is not None:
        return
    with _init_lock:
        if client is None:
            if not MILVUS_PASSWORD:
                raise RuntimeError("MILVUS_PASSWORD is required (set via Kubernetes secret, not ConfigMap)")
            client = MilvusClient(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)


def _search_collection(
    collection_name: str, query: str, top_k: int, output_fields: list[str], filter_expr: str = ""
) -> list[dict]:
    """Encode query via TEI and search Milvus."""
    _init()
    try:
        client.load_collection(collection_name)
    except Exception as e:
        raise RuntimeError(f"Milvus load_collection failed for {collection_name}: {e}") from e

    try:
        embedding = embed_query(query, url=EMBEDDINGS_URL or None)
    except Exception as e:
        raise RuntimeError(f"Embeddings service request failed: {e}") from e

    search_params = {
        "collection_name": collection_name,
        "data": [embedding],
        "limit": top_k,
        "output_fields": output_fields,
    }
    if filter_expr:
        search_params["filter"] = filter_expr
    return client.search(**search_params)[0]


def _safe_filter_value(name: str, value: str) -> str:
    if not _FILTER_VALUE_RE.fullmatch(value):
        raise ValueError(f"Invalid {name} filter value: {value!r}")
    return value


def _search_stems(value: str) -> set[str]:
    """Return lightweight stems for lexical metadata reranking."""
    return {token[:6].lower() for token in _SEARCH_TOKEN_RE.findall(value) if len(token) >= 4}


def _top_document_hit(query: str, hits: list[dict]) -> dict:
    """Choose one document using dense score plus path/URL lexical overlap."""
    query_stems = _search_stems(query)
    candidates = {}
    for rank, hit in enumerate(hits):
        entity = hit.get("entity", {})
        source_key = (entity.get("citation_url", ""), entity.get("file_path", ""))
        if not any(source_key):
            continue
        metadata_stems = _search_stems(" ".join(source_key))
        score = (
            len(query_stems & metadata_stems),
            float(hit.get("distance", 0.0)),
            -rank,
        )
        previous = candidates.get(source_key)
        if previous is None or score > previous[0]:
            candidates[source_key] = (score, hit)
    if not candidates:
        return hits[0]
    return max(candidates.values(), key=lambda candidate: candidate[0])[1]


def _expand_top_document(query: str, hits: list[dict]) -> list[dict]:
    """Replace chunk hits with bounded, ordered context from the best page."""
    if not hits:
        return hits
    selected = _top_document_hit(query, hits)
    selected_entity = selected.get("entity", {})
    file_path = selected_entity.get("file_path", "")
    if not file_path:
        return hits

    try:
        rows = client.query(
            collection_name=COLLECTION_NAME,
            filter=f"file_path == {json.dumps(file_path)}",
            output_fields=["content_text", "citation_url", "file_path", "chunk_index"],
            limit=DOCS_CONTEXT_MAX_CHUNKS,
        )
    except Exception:
        return hits
    if not isinstance(rows, list) or not rows:
        return hits

    unique_rows = {}
    for row in rows:
        content = row.get("content_text", "").strip()
        if content:
            unique_rows.setdefault((int(row.get("chunk_index", 0)), content), row)

    context_chunks = []
    context_chars = 0
    for (_, content), _row in sorted(unique_rows.items(), key=lambda item: item[0][0]):
        separator_chars = 2 if context_chunks else 0
        remaining = DOCS_CONTEXT_MAX_CHARS - context_chars - separator_chars
        if remaining <= 0:
            break
        context_chunks.append(content[:remaining])
        context_chars += min(len(content), remaining) + separator_chars

    if not context_chunks:
        return hits
    expanded_entity = dict(selected_entity)
    expanded_entity["content_text"] = "\n\n".join(context_chunks)
    expanded_entity["citation_url"] = rows[0].get("citation_url", selected_entity.get("citation_url", ""))
    expanded_entity["file_path"] = rows[0].get("file_path", file_path)
    return [{**selected, "entity": expanded_entity}]


@mcp.tool()
def search_kubeflow_docs(query: str, top_k: int = 5) -> str:
    """Search Kubeflow documentation using semantic similarity."""
    try:
        hits = _search_collection(
            COLLECTION_NAME,
            query,
            top_k,
            ["content_text", "citation_url", "file_path", "chunk_index"],
        )
    except RuntimeError as e:
        return f"Search failed: {e}"

    if not hits:
        return "No results found for your query."

    # Dense chunk search often finds the correct page but not the exact section
    # needed for a broad question. Rerank pages using URL/path terms, then give
    # the model bounded, ordered context from that one canonical document.
    hits = _expand_top_document(query, hits)

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
    """Search Kubeflow GitHub issues for bug reports, troubleshooting, and community solutions."""
    filters = []
    if repo:
        repo = _safe_filter_value("repo", repo)
        filters.append(f'repo_name == "{repo}"')
    if state:
        state = _safe_filter_value("state", state)
        filters.append(f'issue_state == "{state}"')
    filter_expr = " and ".join(filters)

    try:
        hits = _search_collection(
            ISSUES_COLLECTION_NAME,
            query,
            top_k,
            ["content_text", "citation_url", "repo_name", "issue_number", "issue_state", "issue_labels"],
            filter_expr=filter_expr,
        )
    except RuntimeError as e:
        return f"Search failed: {e}"

    if not hits:
        return "No issues found for your query."

    results = []
    for i, hit in enumerate(hits, 1):
        entity = hit["entity"]
        entry = f"### Result {i} (score: {hit['distance']:.4f})"
        entry += f"\n**Source:** {entity.get('citation_url', '')}"
        entry += f"\n**Repo:** {entity.get('repo_name', '')}"

        issue_num = entity.get("issue_number", "")
        issue_state = entity.get("issue_state", "")
        labels = entity.get("issue_labels", "")
        if issue_num:
            entry += f"\n**Issue:** #{issue_num}"
        if issue_state:
            entry += f" ({issue_state})"
        if labels:
            entry += f"\n**Labels:** {labels}"

        entry += f"\n\n{entity.get('content_text', '')}\n"
        results.append(entry)

    return "\n---\n".join(results)


@mcp.tool()
def search_kubeflow_code(query: str, top_k: int = 5, resource_kind: str = "") -> str:
    """Search Kubeflow code and YAML manifests using semantic similarity."""
    if resource_kind:
        resource_kind = _safe_filter_value("resource_kind", resource_kind)
    filter_expr = f"resource_kind == '{resource_kind}'" if resource_kind else ""

    try:
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
    except RuntimeError as e:
        return f"Search failed: {e}"

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
