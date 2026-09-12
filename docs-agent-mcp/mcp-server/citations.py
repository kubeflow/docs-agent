import re

from fastmcp.tools import ToolResult

MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(\s*https?://[^)]+\)")
BARE_HTTP_URL = re.compile(r"https?://[^\s<>()]+")


def sanitize_evidence_text(value: object) -> str:
    """Keep URLs out of LLM-facing evidence; structured citations retain them."""
    text = str(value or "")
    text = MARKDOWN_LINK.sub(r"\1", text)
    return BARE_HTTP_URL.sub("", text)


def format_docs_hits(hits: list[dict]) -> tuple[str, list[dict]]:
    results: list[str] = []
    citations: list[dict] = []
    for i, hit in enumerate(hits, 1):
        cid = f"c{i}"
        entity = hit["entity"]
        entry = f"### Result {i} [{cid}] (score: {hit['distance']:.4f})"
        section_path = entity.get("section_path")
        if section_path:
            entry += f"\n**Section:** {section_path}"
        version = entity.get("version")
        if version:
            entry += f"\n**Version:** {version}"
        release_date = entity.get("release_date")
        if release_date is not None:
            entry += f"\n**Release date:** {release_date}"
        entry += f"\n\n{sanitize_evidence_text(entity.get('content_text', ''))}\n"
        results.append(entry)

        citation: dict = {
            "id": cid,
            "url": entity.get("citation_url", ""),
            "score": hit["distance"],
        }
        if section_path:
            citation["section"] = section_path
        if version:
            citation["version"] = version
        if release_date is not None:
            citation["release_date"] = release_date
        doc_type = entity.get("doc_type")
        if doc_type:
            citation["doc_type"] = doc_type
        file_path = entity.get("file_path")
        if file_path:
            citation["file_path"] = file_path
        citations.append(citation)

    return "\n---\n".join(results), citations


def format_issues_hits(hits: list[dict]) -> tuple[str, list[dict]]:
    results: list[str] = []
    citations: list[dict] = []
    for i, hit in enumerate(hits, 1):
        cid = f"c{i}"
        entity = hit["entity"]
        entry = f"### Result {i} [{cid}] (score: {hit['distance']:.4f})"
        repo_name = entity.get("repo_name", "")
        if repo_name:
            entry += f"\n**Repo:** {repo_name}"

        issue_num = entity.get("issue_number", "")
        issue_state = entity.get("issue_state", "")
        labels = entity.get("issue_labels", "")
        if issue_num:
            entry += f"\n**Issue:** #{issue_num}"
        if issue_state:
            entry += f" ({issue_state})"
        if labels:
            entry += f"\n**Labels:** {labels}"

        entry += f"\n\n{sanitize_evidence_text(entity.get('content_text', ''))}\n"
        results.append(entry)

        citation: dict = {
            "id": cid,
            "url": entity.get("citation_url", ""),
            "score": hit["distance"],
        }
        if repo_name:
            citation["repo_name"] = repo_name
        if issue_num:
            citation["issue_number"] = issue_num
        if issue_state:
            citation["issue_state"] = issue_state
        if labels:
            citation["issue_labels"] = labels
        citations.append(citation)

    return "\n---\n".join(results), citations


def format_code_hits(hits: list[dict]) -> tuple[str, list[dict]]:
    results: list[str] = []
    citations: list[dict] = []
    for i, hit in enumerate(hits, 1):
        cid = f"c{i}"
        entity = hit["entity"]
        entry = f"### Result {i} [{cid}] (score: {hit['distance']:.4f})"

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

        entry += f"\n\n```\n{sanitize_evidence_text(entity.get('content_text', ''))}\n```\n"
        results.append(entry)

        citation: dict = {
            "id": cid,
            "url": entity.get("citation_url", ""),
            "score": hit["distance"],
        }
        file_path = entity.get("file_path")
        if file_path:
            citation["file_path"] = file_path
        if kind:
            citation["resource_kind"] = kind
        if name:
            citation["resource_name"] = name
        if ns:
            citation["resource_namespace"] = ns
        if ftype:
            citation["file_type"] = ftype
        citations.append(citation)

    return "\n---\n".join(results), citations


def text_tool_result(message: str, *, is_error: bool = False) -> ToolResult:
    return ToolResult(content=message, is_error=is_error)


def search_tool_result(
    body: str,
    citations: list[dict],
    retrieval: dict | None = None,
) -> ToolResult:
    structured: dict = {"citations": citations}
    if retrieval is not None:
        structured["retrieval"] = retrieval
    return ToolResult(content=body, structured_content=structured)
