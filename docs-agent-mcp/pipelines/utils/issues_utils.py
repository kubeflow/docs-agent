"""Utility functions for GitHub issues pipeline chunking and metadata extraction.

download_github_issues emits JSONL records that carry both:
- content: human-readable markdown (for RAG embedding text)
- structured fields (title, repo_name, issue_number, …, body, comments):
  machine-readable metadata so downstream steps do not regex-parse MD
"""

from __future__ import annotations

import re
from typing import Any


def format_issue_markdown(
    title: str,
    repo_name: str,
    issue_number: int,
    url: str,
    labels: str,
    state: str,
    created_at: str,
    updated_at: str,
    body: str,
    comments: list[dict[str, Any]] | None = None,
) -> str:
    """Render the human-readable markdown form of an issue (plus comments)."""
    content = f"# {title}\n\n"
    content += f"**Repository:** {repo_name}\n"
    content += f"**Issue:** #{issue_number}\n"
    content += f"**URL:** {url}\n"
    content += f"**Labels:** {labels}\n"
    content += f"**State:** {state}\n"
    content += f"**Created:** {created_at}\n"
    content += f"**Updated:** {updated_at}\n\n"
    content += body or ""

    for comment in comments or []:
        author = comment.get("author", "unknown")
        created = comment.get("created_at", "")
        cbody = comment.get("body", "") or ""
        content += f"\n\n---\n**Comment by @{author}** ({created}):\n{cbody}"

    return content


def build_issue_record(
    *,
    repo_name: str,
    repo_short_name: str,
    issue_number: int,
    title: str,
    url: str,
    labels: str,
    state: str,
    created_at: str,
    updated_at: str,
    body: str,
    comments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a download_github_issues JSONL record (markdown + structured fields)."""
    comments = comments or []
    content = format_issue_markdown(
        title=title,
        repo_name=repo_name,
        issue_number=issue_number,
        url=url,
        labels=labels,
        state=state,
        created_at=created_at,
        updated_at=updated_at,
        body=body,
        comments=comments,
    )
    return {
        # Human-readable artifact (RAG embedding source text)
        "path": f"issues/{repo_short_name}/{issue_number}",
        "content": content,
        "file_name": f"issue-{repo_short_name}-{issue_number}.md",
        "url": url,
        # Machine-readable fields (preferred by chunk_and_embed_issues)
        "title": title,
        "repo_name": repo_name,
        "issue_number": int(issue_number),
        "issue_state": state,
        "issue_labels": labels,
        "created_at": created_at,
        "updated_at": updated_at,
        "body": body or "",
        "comments": comments,
    }


def _is_empty_metadata(metadata: dict[str, Any]) -> bool:
    return (
        not metadata.get("title")
        and not metadata.get("repo_name")
        and not metadata.get("issue_state")
        and not metadata.get("issue_labels")
        and not metadata.get("citation_url")
        and int(metadata.get("issue_number") or 0) == 0
    )


def has_structured_metadata(record: dict[str, Any]) -> bool:
    """True when the record carries machine-readable issue fields."""
    if not isinstance(record, dict):
        return False
    if "issue_number" in record and record.get("issue_number") not in (None, ""):
        return True
    return bool(record.get("title") or record.get("repo_name") or record.get("issue_state"))


def metadata_from_structured(record: dict[str, Any]) -> dict[str, Any]:
    """Map structured download fields onto the chunking metadata schema."""
    issue_number = record.get("issue_number", 0)
    try:
        issue_number = int(issue_number or 0)
    except (TypeError, ValueError):
        issue_number = 0

    return {
        "title": (record.get("title") or "").strip(),
        "repo_name": (record.get("repo_name") or "").strip(),
        "issue_number": issue_number,
        "issue_state": (record.get("issue_state") or "").strip(),
        "issue_labels": (record.get("issue_labels") or "").strip(),
        "citation_url": (
            record.get("url")
            or record.get("citation_url")
            or ""
        ).strip(),
    }


def parse_issue_metadata(content: str) -> dict:
    """Extract structured metadata from download_github_issues markdown content.

    Legacy fallback for records that only have the human-readable `content`
    field. Prefer resolve_issue_metadata() on full JSONL records.

    Args:
        content: Raw issue content string from download_github_issues.

    Returns:
        Dict with keys: title, repo_name, issue_number, issue_state,
        issue_labels, citation_url. Missing fields default to empty string
        (or 0 for issue_number).
    """
    title_match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
    repo_match = re.search(r'\*\*Repository:\*\*\s*(.+)', content)
    number_match = re.search(r'\*\*Issue:\*\*\s*#(\d+)', content)
    url_match = re.search(r'\*\*URL:\*\*\s*(.+)', content)
    labels_match = re.search(r'\*\*Labels:\*\*[ \t]*(.*)', content)
    state_match = re.search(r'\*\*State:\*\*\s*(\w+)', content)

    return {
        "title": title_match.group(1).strip() if title_match else "",
        "repo_name": repo_match.group(1).strip() if repo_match else "",
        "issue_number": int(number_match.group(1)) if number_match else 0,
        "issue_state": state_match.group(1).strip() if state_match else "",
        "issue_labels": labels_match.group(1).strip() if labels_match else "",
        "citation_url": url_match.group(1).strip() if url_match else "",
    }


def resolve_issue_metadata(record: dict[str, Any] | str) -> dict[str, Any]:
    """Resolve issue metadata, preferring structured JSON fields over MD regex.

    Args:
        record: A download_github_issues JSONL dict, or a legacy markdown string.

    Returns:
        Metadata dict with title/repo_name/issue_number/issue_state/
        issue_labels/citation_url.
    """
    if isinstance(record, str):
        metadata = parse_issue_metadata(record)
        if _is_empty_metadata(metadata):
            print(
                "WARNING: Failed to parse GitHub issue metadata. "
                "Upstream markdown format may have changed."
            )
        return metadata

    if has_structured_metadata(record):
        return metadata_from_structured(record)

    metadata = parse_issue_metadata(record.get("content", "") or "")
    if _is_empty_metadata(metadata):
        print(
            "WARNING: Failed to parse GitHub issue metadata. "
            "Upstream markdown format may have changed."
        )
    return metadata


def build_metadata_prefix(metadata: dict) -> str:
    """Build a self-contained prefix string for each chunk.

    Args:
        metadata: Dict from parse_issue_metadata / resolve_issue_metadata.

    Returns:
        Prefix string like "[Issue #42] Title | Repo: x | State: open | Labels: y"
    """
    parts = [f"[Issue #{metadata['issue_number']}] {metadata['title']}"]
    if metadata["repo_name"]:
        parts.append(f"Repo: {metadata['repo_name']}")
    if metadata["issue_state"]:
        parts.append(f"State: {metadata['issue_state']}")
    if metadata["issue_labels"]:
        parts.append(f"Labels: {metadata['issue_labels']}")
    return " | ".join(parts) + "\n\n"


def _comment_segment(comment: dict[str, Any]) -> str:
    author = comment.get("author", "unknown")
    created = comment.get("created_at", "")
    body = comment.get("body", "") or ""
    return f"**Comment by @{author}** ({created}):\n{body}"


def issue_content_segments(record: dict[str, Any]) -> list[str]:
    """Return body/comment segments for chunking without regex-splitting markdown.

    Prefers structured `body` + `comments` when present; otherwise falls back
    to splitting the human-readable `content` on comment separators.
    """
    if isinstance(record.get("body"), str) and (
        "comments" in record or has_structured_metadata(record)
    ):
        segments: list[str] = []
        # Include title + body as the first segment (matches prior MD layout)
        title = (record.get("title") or "").strip()
        body = record.get("body") or ""
        header_bits = []
        if title:
            header_bits.append(f"# {title}")
        if record.get("repo_name"):
            header_bits.append(f"**Repository:** {record['repo_name']}")
        if record.get("issue_number"):
            header_bits.append(f"**Issue:** #{record['issue_number']}")
        url = record.get("url") or record.get("citation_url") or ""
        if url:
            header_bits.append(f"**URL:** {url}")
        if record.get("issue_labels") is not None:
            header_bits.append(f"**Labels:** {record.get('issue_labels') or ''}")
        if record.get("issue_state"):
            header_bits.append(f"**State:** {record['issue_state']}")
        if record.get("created_at"):
            header_bits.append(f"**Created:** {record['created_at']}")
        if record.get("updated_at"):
            header_bits.append(f"**Updated:** {record['updated_at']}")

        first = "\n".join(header_bits)
        if first and body:
            first = f"{first}\n\n{body}"
        elif body:
            first = body
        if first.strip():
            segments.append(first.strip())

        for comment in record.get("comments") or []:
            seg = _comment_segment(comment).strip()
            if seg:
                segments.append(seg)
        return segments if segments else [record.get("content", "") or ""]

    content = record.get("content", "") if isinstance(record, dict) else str(record)
    return [s.strip() for s in re.split(r"\n\n---\n", content or "") if s.strip()] or [content or ""]


def split_issue_into_chunks(
    content: str,
    metadata_prefix: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
) -> list[str]:
    """Split issue content into chunks at comment boundaries.

    Strategy:
    - Split at '\\n\\n---\\n' boundaries (comment separators from download_github_issues)
    - If entire content fits in one chunk, return single chunk
    - If a segment exceeds chunk_size, subdivide with RecursiveCharacterTextSplitter
    - Prepend metadata_prefix to every chunk

    Args:
        content: Full issue content string.
        metadata_prefix: Prefix to prepend to each chunk.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between subdivided chunks.

    Returns:
        List of chunk strings, each prefixed with metadata.
    """
    return split_segments_into_chunks(
        segments=re.split(r"\n\n---\n", content),
        metadata_prefix=metadata_prefix,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        fallback_content=content,
    )


def split_segments_into_chunks(
    segments: list[str],
    metadata_prefix: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
    fallback_content: str = "",
) -> list[str]:
    """Chunk pre-split body/comment segments, prefixing each chunk."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    effective_size = chunk_size - len(metadata_prefix)
    if effective_size < 100:
        effective_size = 100

    joined = "\n\n".join(s.strip() for s in segments if s and s.strip())
    if len(joined) <= effective_size:
        source = joined or fallback_content
        return [metadata_prefix + source]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[str] = []
    for segment in segments:
        segment = (segment or "").strip()
        if not segment:
            continue
        if len(segment) <= effective_size:
            chunks.append(metadata_prefix + segment)
        else:
            for sub in splitter.split_text(segment):
                chunks.append(metadata_prefix + sub)

    if chunks:
        return chunks
    source = joined or fallback_content
    return [metadata_prefix + source[:effective_size]]


def split_issue_record_into_chunks(
    record: dict[str, Any],
    metadata_prefix: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
) -> list[str]:
    """Chunk a download_github_issues record using structured segments when available."""
    segments = issue_content_segments(record)
    return split_segments_into_chunks(
        segments=segments,
        metadata_prefix=metadata_prefix,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        fallback_content=record.get("content", "") if isinstance(record, dict) else "",
    )
