"""Utility functions for GitHub issues pipeline chunking and metadata extraction."""

import re


def parse_issue_metadata(content: str) -> dict:
    """Extract structured metadata from download_github_issues output format.

    The content follows the format produced by the download_github_issues
    KFP component: title line, metadata lines, body, and comments.

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


def build_metadata_prefix(metadata: dict) -> str:
    """Build a self-contained prefix string for each chunk.

    Args:
        metadata: Dict from parse_issue_metadata.

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
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Available space for content after prefix
    effective_size = chunk_size - len(metadata_prefix)
    if effective_size < 100:
        effective_size = 100

    # If entire content fits, return single chunk
    if len(content) <= effective_size:
        return [metadata_prefix + content]

    # Split at comment boundaries
    segments = re.split(r'\n\n---\n', content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        if len(segment) <= effective_size:
            chunks.append(metadata_prefix + segment)
        else:
            # Subdivide oversized segment
            sub_chunks = splitter.split_text(segment)
            for sub in sub_chunks:
                chunks.append(metadata_prefix + sub)

    return chunks if chunks else [metadata_prefix + content[:effective_size]]
