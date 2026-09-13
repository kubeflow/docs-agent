"""MCP tools: search docs, issues, and code."""

import os
import re

from fastmcp import FastMCP
from fastmcp.tools import ToolResult

from citations import (
    format_code_hits,
    format_docs_hits,
    format_issues_hits,
    search_tool_result,
    text_tool_result,
)
from intent_router import RetrievalPlan, retrieval_metadata
import milvus_search
from milvus_search import search_collection, search_docs_auto

PORT = int(os.getenv("PORT", "8000"))
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "512"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))

SAFE_FILTER = re.compile(r"^[A-Za-z0-9_/.\-]+$")

DOCS_OUTPUT_FIELDS = [
    "content_text",
    "citation_url",
    "file_path",
    "release_date",
    "doc_type",
    "version",
    "section_path",
]

mcp = FastMCP("Kubeflow Docs MCP Server")


def _safe_filter_value(name: str, value: str) -> str:
    if not SAFE_FILTER.fullmatch(value):
        raise ValueError(f"Invalid {name} filter value: {value!r}")
    return value


def _search_args(query: str, top_k: int) -> tuple[str, int]:
    """Normalize bounded tool arguments before spending embedding/vector resources."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    query = " ".join(query.split())
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query exceeds the {MAX_QUERY_CHARS}-character limit")
    try:
        top_k = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be an integer") from exc
    return query, min(MAX_TOP_K, max(1, top_k))


@mcp.tool()
def search_kubeflow_docs(query: str, top_k: int = 5) -> ToolResult:
    """Search Kubeflow documentation. Search mode is chosen here, not by the LLM."""
    try:
        query, top_k = _search_args(query, top_k)
    except ValueError as exc:
        return text_tool_result(f"Search rejected: {exc}")
    try:
        if milvus_search.SEARCH_MODE == "auto":
            hits, retrieval_meta = search_docs_auto(query, top_k, DOCS_OUTPUT_FIELDS)
        else:
            hits = search_collection(
                milvus_search.COLLECTION_NAME, query, top_k, DOCS_OUTPUT_FIELDS
            )
            retrieval_meta = retrieval_metadata(
                RetrievalPlan(
                    intent="explicit",
                    retrieval_mode=milvus_search.SEARCH_MODE,
                    reason=f"SEARCH_MODE={milvus_search.SEARCH_MODE}",
                )
            )
    except RuntimeError as exc:
        return text_tool_result(f"Search failed: {exc}")

    if not hits:
        return text_tool_result("No results found for your query.")

    body, citations = format_docs_hits(hits)
    return search_tool_result(body, citations, retrieval=retrieval_meta)


@mcp.tool()
def search_github_issues(query: str, top_k: int = 5, repo: str = "", state: str = "") -> ToolResult:
    """Search Kubeflow GitHub issues."""
    try:
        query, top_k = _search_args(query, top_k)
    except ValueError as exc:
        return text_tool_result(f"Search rejected: {exc}")
    filters = []
    if repo:
        repo = _safe_filter_value("repo", repo)
        filters.append(f'repo_name == "{repo}"')
    if state:
        state = _safe_filter_value("state", state)
        filters.append(f'issue_state == "{state}"')
    filter_expr = " and ".join(filters)

    try:
        hits = search_collection(
            milvus_search.ISSUES_COLLECTION_NAME,
            query,
            top_k,
            ["content_text", "citation_url", "repo_name", "issue_number", "issue_state", "issue_labels"],
            filter_expr=filter_expr,
        )
    except RuntimeError as exc:
        return text_tool_result(f"Search failed: {exc}")

    if not hits:
        return text_tool_result("No issues found for your query.")

    body, citations = format_issues_hits(hits)
    return search_tool_result(body, citations)


@mcp.tool()
def search_kubeflow_code(
    query: str, top_k: int = 5, resource_kind: str = "", repo: str = ""
) -> ToolResult:
    """Search Kubeflow code and YAML manifests."""
    try:
        query, top_k = _search_args(query, top_k)
    except ValueError as exc:
        return text_tool_result(f"Search rejected: {exc}")
    filters = []
    if resource_kind:
        resource_kind = _safe_filter_value("resource_kind", resource_kind)
        filters.append(f"resource_kind == '{resource_kind}'")
    if repo:
        repo = _safe_filter_value("repo", repo)
        filters.append(f'repo_name == "{repo}"')
    filter_expr = " and ".join(filters)

    try:
        hits = search_collection(
            milvus_search.CODE_COLLECTION_NAME,
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
    except RuntimeError as exc:
        return text_tool_result(f"Search failed: {exc}")

    if not hits:
        return text_tool_result("No code results found for your query.")

    body, citations = format_code_hits(hits)
    return search_tool_result(body, citations)


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=PORT)
