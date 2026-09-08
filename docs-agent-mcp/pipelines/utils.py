"""Shared utility functions for pipeline components."""

from __future__ import annotations

import os
import re
from typing import Sequence

import requests

# Milvus collection names (underscores only — hyphens are invalid in Milvus).
DOCS_COLLECTION = "kubeflow_docs"
ISSUES_COLLECTION = "issues_rag"
CODE_COLLECTION = "code_rag"

DEFAULT_EMBEDDINGS_URL = (
    "http://embeddings-service-predictor.ml-infra.svc.cluster.local/embed"
)
DEFAULT_MILVUS_HOST = "milvus-milvus.ml-infra.svc.cluster.local"
# Hugo content root the docs pipelines index.
DOCS_CONTENT_ROOT = "content/en/docs"
# all-mpnet-base-v2 output size; other TEI models may differ — pass embedding_dim.
DEFAULT_EMBEDDING_DIM = 768
# Docs defaults: YAML-heavy pages tokenize denser than prose, so keep TEI inputs
# well under the 384-token ceiling (1000 chars produced HTTP 413 on Katib docs).
DEFAULT_DOCS_CHUNK_SIZE = 600
DEFAULT_DOCS_CHUNK_OVERLAP = 60
DEFAULT_DOCS_MAX_TEI_CHARS = 600
# Generic TEI ceiling used by non-docs pipelines / helpers.
MAX_TEI_INPUT_CHARS = 1000
# Batch count only; per-input size is limited by max_tei_chars / MAX_TEI_INPUT_CHARS.
DEFAULT_EMBEDDING_BATCH_SIZE = 8


def truncate_for_tei(text: str, max_chars: int = MAX_TEI_INPUT_CHARS) -> str:
    """Truncate text so TEI accepts it (413 if any input exceeds token limit)."""
    if not text:
        return ""
    return text[:max_chars]


def resolve_github_token(github_token: str = "") -> str:
    """Resolve a GitHub PAT from the pipeline parameter or environment.

    Checks ``github_token`` first, then ``Github_Pat`` (repo/OKE secret name),
    then ``GITHUB_TOKEN`` for compatibility with other tooling.
    """
    for candidate in (
        github_token,
        os.environ.get("Github_Pat", ""),
        os.environ.get("GITHUB_TOKEN", ""),
    ):
        if candidate and candidate.strip():
            return candidate.strip()
    return ""


def clean_content(content: str) -> str:
    """Clean raw document content for better embeddings.

    Removes Hugo frontmatter, template syntax, HTML tags, navigation
    artifacts and URLs, keeping the line structure that makes code and
    YAML retrievable.

    Args:
        content: Raw document content (markdown/HTML).

    Returns:
        Cleaned text suitable for embedding.
    """
    # Remove Hugo frontmatter (both --- and +++ styles)
    # \A anchors to absolute start of string; backreference ensures matching delimiters
    content = re.sub(
        r'\A[ \t]*(?P<delimiter>---|\+\+\+)[ \t]*\r?\n.*?'
        r'^[ \t]*(?P=delimiter)[ \t]*(?:\r?\n|\Z)',
        '', content,
        flags=re.DOTALL | re.MULTILINE
    )

    # Remove Hugo template syntax
    content = re.sub(r'\{\{.*?\}\}', '', content, flags=re.DOTALL)

    # Remove HTML comments and tags
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    content = re.sub(r'<[^>]+>', ' ', content)

    # Remove navigation/menu artifacts
    content = re.sub(
        r'\b(Get Started|Contribute|GenAI|Home|Menu|Navigation)\b', '',
        content, flags=re.IGNORECASE
    )

    # Convert Markdown links before removing bare URLs. Doing this in
    # the reverse order leaves dangling `](` tokens; the link regex can
    # then span multiple paragraphs and delete intervening YAML.
    content = re.sub(r'\[([^\]]+)\]\((?:[^()]|\([^()]*\))*\)', r'\1', content)
    content = re.sub(r'https?://[^\s]+', '', content)

    # Collapse horizontal whitespace and blank-line runs, keeping newlines
    content = re.sub(r'[ \t]+', ' ', content)
    content = re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+', '\n\n', content)
    content = content.strip()

    return content


def build_docs_citation_url(file_path: str, base_url: str) -> str:
    base = base_url.rstrip('/')
    marker_index = file_path.find(DOCS_CONTENT_ROOT)
    if marker_index == -1:
        return f"{base}/{file_path}"

    url_path = os.path.splitext(file_path[marker_index + len(DOCS_CONTENT_ROOT):].strip('/'))[0]
    parent, _, page_name = url_path.rpartition('/')
    if page_name in ('_index', 'index'):
        url_path = parent
    return f"{base}/{url_path}/" if url_path else f"{base}/"


def embed_texts(
    texts: Sequence[str],
    embeddings_service_url: str,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    timeout: int = 120,
) -> list[list[float]]:
    """Call the in-cluster TEI embeddings service for a list of texts."""
    if not texts:
        return []
    if not embeddings_service_url.strip():
        raise ValueError("embeddings_service_url is required")

    batch_size = max(1, int(batch_size))
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = [truncate_for_tei(t) for t in texts[start : start + batch_size]]
        response = requests.post(
            embeddings_service_url.strip(),
            json={"inputs": batch},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) != len(batch):
            raise RuntimeError(
                f"Embeddings service returned unexpected payload for batch size {len(batch)}"
            )
        vectors.extend(payload)

    return vectors
