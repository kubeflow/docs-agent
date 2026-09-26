"""Canonical parsing and token-aware chunking for the production RAG v4 pipeline.

Parses Hugo/Markdown documentation into structured sections and emits chunk
records compatible with the kubeflow_docs Milvus schema, plus parser/chunker
version metadata for downstream workers.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from hugo_ingest import parse_frontmatter, process_html_table

PARSER_VERSION = "1.0.0"
CHUNKER_VERSION = "1.0.0"
SECTION_PATH_SEP = " > "
DEFAULT_TARGET_TOKENS = 350
DEFAULT_OVERLAP_TOKENS = 50
MAX_CONTENT_TEXT_CHARS = 2000
CHARS_PER_TOKEN = 4

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE_START_RE = re.compile(r"^(`{3,}|~{3,})(\w*)\s*$")
GFM_TABLE_ROW_RE = re.compile(r"^\s*\|")
ALERT_RE = re.compile(
    r"\{\{%\s*alert\b[^%]*%\}\}(.*?)\{\{%\s*/alert\s*%\}\}",
    re.DOTALL | re.IGNORECASE,
)
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
HUGO_SHORTCODE_RE = re.compile(r"\{\{.*?%\}\}|\{\{.*?\}\}", re.DOTALL)
IMG_TAG_RE = re.compile(r'<img[^>]*alt="([^"]*)"[^>]*>', re.IGNORECASE)
MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RELEASE_DATE_HTML_RE = re.compile(
    r"<th[^>]*>\s*Release Date\s*</th>\s*<td>\s*(\d{4}-\d{2}-\d{2})\s*</td>",
    re.IGNORECASE | re.DOTALL,
)
RELEASE_DATE_GFM_RE = re.compile(
    r"^\s*\|\s*Release Date\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|",
    re.MULTILINE | re.IGNORECASE,
)
FRONTMATTER_RELEASE_DATE_KEYS = ("release_date", "releaseDate", "ga_date")


@dataclass
class LinkRef:
    text: str
    url: str

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "url": self.url}


@dataclass
class CanonicalBlock:
    block_type: str
    content: str
    links: list[LinkRef] = field(default_factory=list)
    language: str = ""
    admonition_type: str = ""
    table_format: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "block_type": self.block_type,
            "content": self.content,
        }
        if self.links:
            payload["links"] = [link.to_dict() for link in self.links]
        if self.language:
            payload["language"] = self.language
        if self.admonition_type:
            payload["admonition_type"] = self.admonition_type
        if self.table_format:
            payload["table_format"] = self.table_format
        return payload


@dataclass
class CanonicalSection:
    heading: str
    heading_level: int
    section_path: str
    blocks: list[CanonicalBlock] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "heading_level": self.heading_level,
            "section_path": self.section_path,
            "blocks": [block.to_dict() for block in self.blocks],
        }


@dataclass
class CanonicalDocument:
    parser_version: str
    title: str
    description: str
    weight: int
    frontmatter: dict[str, Any]
    sections: list[CanonicalSection]
    links: list[LinkRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser_version": self.parser_version,
            "title": self.title,
            "description": self.description,
            "weight": self.weight,
            "frontmatter": self.frontmatter,
            "sections": [section.to_dict() for section in self.sections],
            "links": [link.to_dict() for link in self.links],
        }


def estimate_tokens(text: str) -> int:
    """Lightweight token estimate aligned with TEI char limits (~4 chars/token)."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def _normalize_heading_text(raw: str) -> str:
    text = raw.strip()
    text = LINK_RE.sub(r"\1", text)
    text = INLINE_CODE_RE.sub(lambda m: m.group(0), text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_links(text: str) -> tuple[str, list[LinkRef]]:
    links: list[LinkRef] = []

    def repl(match: re.Match[str]) -> str:
        links.append(LinkRef(text=match.group(1), url=match.group(2)))
        return match.group(0)

    preserved = LINK_RE.sub(repl, text)
    return preserved, links


def _collapse_horizontal_whitespace(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lines.append(re.sub(r"[ \t]+", " ", line).rstrip())
    return "\n".join(lines).strip()


def _alert_label(raw_attrs: str, inner: str) -> str:
    title_match = re.search(r'title="([^"]+)"', raw_attrs, re.IGNORECASE)
    color_match = re.search(r'color="([^"]+)"', raw_attrs, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else ""
    color = color_match.group(1).strip().lower() if color_match else ""
    if title:
        label = title.upper().rstrip(":")
    elif color in {"warning", "danger"}:
        label = "WARNING"
    else:
        label = "NOTE"
    return f"{label}: {inner.strip()}"


def _expand_admonitions(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        full = match.group(0)
        attrs_match = re.match(r"\{\{%\s*alert\b([^%]*)\%\}\}", full, re.IGNORECASE)
        attrs = attrs_match.group(1) if attrs_match else ""
        return _alert_label(attrs, match.group(1))

    return ALERT_RE.sub(repl, text)


def _strip_hugo_artifacts(text: str) -> str:
    text = HUGO_SHORTCODE_RE.sub("", text)
    text = text.replace("fa-check", "yes").replace("fa-xmark", "no")
    return text


def _stash_for_html(text: str) -> str:
    """Protect code and tables before HTML parsing (see hugo_ingest ordering)."""
    stashes: dict[str, str] = {}

    def stash(prefix: str, match: re.Match[str]) -> str:
        key = f"%%{prefix}{len(stashes)}%%"
        stashes[key] = match.group(0)
        return key

    text = re.sub(r"```.*?```", lambda m: stash("FENCE", m), text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]+`", lambda m: stash("CODE", m), text)
    text = re.sub(r"(?:\|.*\|[\r\n]+)+", lambda m: stash("GFM", m), text)

    if "<table" in text:
        text = process_html_table(text)
    text = IMG_TAG_RE.sub(r"Figure: \1", text)
    text = MD_IMAGE_RE.sub(r"Figure: \1", text)

    if "<" in text and ">" in text:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator="\n", strip=False)

    for key, value in stashes.items():
        text = text.replace(key, value)
    return text


def _build_section_path(title: str, heading_stack: dict[int, str]) -> str:
    parts: list[str] = []
    if title:
        parts.append(title)
    for level in sorted(heading_stack):
        parts.append(heading_stack[level])
    return SECTION_PATH_SEP.join(parts)


def _flush_prose(
    buffer: list[str],
    section: CanonicalSection,
    doc_links: list[LinkRef],
) -> None:
    if not buffer:
        return
    raw = _collapse_horizontal_whitespace("\n".join(buffer))
    if not raw:
        buffer.clear()
        return
    content, links = _extract_links(raw)
    doc_links.extend(links)
    section.blocks.append(
        CanonicalBlock(block_type="prose", content=content, links=links)
    )
    buffer.clear()


def _parse_gfm_table(lines: Sequence[str], start: int) -> tuple[list[str], int]:
    table_lines: list[str] = []
    idx = start
    while idx < len(lines) and GFM_TABLE_ROW_RE.match(lines[idx]):
        table_lines.append(lines[idx].rstrip())
        idx += 1
    return table_lines, idx


def _table_header_and_rows(table_text: str) -> tuple[str, str, list[str]]:
    rows = [line for line in table_text.splitlines() if line.strip()]
    if not rows:
        return "", "", []
    header = rows[0]
    if len(rows) > 1 and re.match(r"^\s*\|?\s*:?-+", rows[1]):
        separator = rows[1]
        body_rows = rows[2:]
    else:
        separator = "| --- |"
        body_rows = rows[1:]
    return header, separator, body_rows


def parse_canonical_document(content: str) -> CanonicalDocument:
    """Parse Markdown/Hugo into a canonical, JSON-serializable document tree."""
    frontmatter, body = parse_frontmatter(content)
    title = str(frontmatter.get("title") or "").strip()
    description = str(frontmatter.get("description") or "").strip()
    weight = int(frontmatter.get("weight") or 0)

    body = _expand_admonitions(body)
    body = _strip_hugo_artifacts(body)
    body = _stash_for_html(body)

    doc_links: list[LinkRef] = []
    sections: list[CanonicalSection] = []
    heading_stack: dict[int, str] = {}
    current_section = CanonicalSection(
        heading="",
        heading_level=0,
        section_path=_build_section_path(title, heading_stack),
    )
    prose_buffer: list[str] = []

    lines = body.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        heading_match = HEADING_RE.match(line)
        if heading_match:
            _flush_prose(prose_buffer, current_section, doc_links)
            if current_section.blocks or current_section.heading or not sections:
                sections.append(current_section)

            level = len(heading_match.group(1))
            heading_text = _normalize_heading_text(heading_match.group(2))
            heading_stack = {lvl: txt for lvl, txt in heading_stack.items() if lvl < level}
            heading_stack[level] = heading_text
            current_section = CanonicalSection(
                heading=heading_text,
                heading_level=level,
                section_path=_build_section_path(title, heading_stack),
            )
            idx += 1
            continue

        fence_match = FENCE_START_RE.match(line.strip())
        if fence_match:
            _flush_prose(prose_buffer, current_section, doc_links)
            fence = fence_match.group(1)
            language = fence_match.group(2) or ""
            fence_char = fence[0]
            fence_len = len(fence)
            block_lines = [line]
            idx += 1
            closed = False
            while idx < len(lines):
                block_lines.append(lines[idx])
                if lines[idx].strip().startswith(fence_char * fence_len):
                    closed = True
                    idx += 1
                    break
                idx += 1
            if not closed:
                idx = len(lines)
            fence_text = "\n".join(block_lines)
            current_section.blocks.append(
                CanonicalBlock(
                    block_type="code_fence",
                    content=fence_text,
                    language=language,
                )
            )
            continue

        if GFM_TABLE_ROW_RE.match(line):
            _flush_prose(prose_buffer, current_section, doc_links)
            table_lines, idx = _parse_gfm_table(lines, idx)
            table_text = "\n".join(table_lines)
            current_section.blocks.append(
                CanonicalBlock(
                    block_type="table",
                    content=table_text,
                    table_format="gfm",
                )
            )
            continue

        prose_buffer.append(line)
        idx += 1

    _flush_prose(prose_buffer, current_section, doc_links)
    if current_section.blocks or current_section.heading or not sections:
        sections.append(current_section)

    if not sections:
        sections = [
            CanonicalSection(
                heading="",
                heading_level=0,
                section_path=_build_section_path(title, heading_stack),
            )
        ]

    return CanonicalDocument(
        parser_version=PARSER_VERSION,
        title=title,
        description=description,
        weight=weight,
        frontmatter=frontmatter,
        sections=sections,
        links=doc_links,
    )


def split_prose_by_tokens(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split prose into token-bounded chunks with overlap."""
    if not text:
        return []
    if estimate_tokens(text) <= target_tokens:
        return [text]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return [text[: MAX_CONTENT_TEXT_CHARS]]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        chunk = "\n\n".join(current).strip()
        if chunk:
            chunks.append(chunk)
        current = []
        current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        if paragraph_tokens > target_tokens:
            flush()
            chunks.extend(
                _split_long_paragraph(paragraph, target_tokens, overlap_tokens)
            )
            continue

        if current_tokens + paragraph_tokens > target_tokens and current:
            flush()
            if chunks and overlap_tokens > 0:
                overlap_text = _tail_tokens(chunks[-1], overlap_tokens)
                if overlap_text:
                    current = [overlap_text, paragraph]
                    current_tokens = estimate_tokens("\n\n".join(current))
                    continue

        current.append(paragraph)
        current_tokens += paragraph_tokens

    flush()
    return chunks or [text[: MAX_CONTENT_TEXT_CHARS]]


def _split_long_paragraph(text: str, target_tokens: int, overlap_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return [text[: MAX_CONTENT_TEXT_CHARS]]

    target_words = max(1, target_tokens * CHARS_PER_TOKEN // 5)
    overlap_words = max(0, overlap_tokens * CHARS_PER_TOKEN // 5)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + target_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(start + 1, end - overlap_words)
    return chunks


def _tail_tokens(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    max_chars = overlap_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[-max_chars:].lstrip()


def _split_table_by_tokens(table_text: str, target_tokens: int) -> list[str]:
    header, separator, body_rows = _table_header_and_rows(table_text)
    if not body_rows:
        return [table_text]

    chunks: list[str] = []
    current_rows: list[str] = []

    for row in body_rows:
        candidate_rows = current_rows + [row]
        candidate = "\n".join([header, separator, *candidate_rows])
        if current_rows and estimate_tokens(candidate) > target_tokens:
            chunks.append("\n".join([header, separator, *current_rows]))
            current_rows = [row]
        else:
            current_rows = candidate_rows

    if current_rows:
        chunks.append("\n".join([header, separator, *current_rows]))

    return chunks or [table_text]


def _chunk_block(
    block: CanonicalBlock,
    section: CanonicalSection,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, str, list[LinkRef]]]:
    """Return (chunk_type, content_text, links) tuples for one canonical block."""
    heading_prefix = ""
    if section.heading:
        heading_prefix = f"{section.section_path}\n\n"

    if block.block_type == "code_fence":
        content = block.content.strip()
        if len(content) > MAX_CONTENT_TEXT_CHARS:
            content = content[:MAX_CONTENT_TEXT_CHARS]
        return [("code", content, block.links)]

    if block.block_type == "table":
        pieces = _split_table_by_tokens(block.content, target_tokens)
        if len(pieces) == 1 and estimate_tokens(pieces[0]) <= target_tokens:
            content = pieces[0]
            if len(content) > MAX_CONTENT_TEXT_CHARS:
                content = content[: MAX_CONTENT_TEXT_CHARS]
            return [("table", content, block.links)]

        rows: list[tuple[str, str, list[LinkRef]]] = []
        for piece in pieces:
            content = piece
            if len(content) > MAX_CONTENT_TEXT_CHARS:
                content = content[: MAX_CONTENT_TEXT_CHARS]
            rows.append(("table_row", content, block.links))
        return rows

    if block.block_type == "prose":
        prose_chunks = split_prose_by_tokens(
            block.content,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        output: list[tuple[str, str, list[LinkRef]]] = []
        for piece in prose_chunks:
            content = heading_prefix + piece if heading_prefix else piece
            if len(content) > MAX_CONTENT_TEXT_CHARS:
                content = content[: MAX_CONTENT_TEXT_CHARS]
            chunk_type = "admonition" if content.lstrip().startswith(("NOTE:", "WARNING:")) else "text"
            output.append((chunk_type, content, block.links))
        return output

    content = block.content.strip()
    if len(content) > MAX_CONTENT_TEXT_CHARS:
        content = content[: MAX_CONTENT_TEXT_CHARS]
    chunk_type = "admonition" if block.block_type == "admonition" else "text"
    return [(chunk_type, content, block.links)]


def chunk_canonical_document(
    document: CanonicalDocument | dict[str, Any],
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    """Section-first chunking with token targets and overlap."""
    if isinstance(document, dict):
        sections = document.get("sections", [])
    else:
        sections = [section.to_dict() for section in document.sections]

    chunks: list[dict[str, Any]] = []
    for section_data in sections:
        section = CanonicalSection(
            heading=section_data.get("heading", ""),
            heading_level=int(section_data.get("heading_level", 0)),
            section_path=section_data.get("section_path", ""),
            blocks=[
                CanonicalBlock(
                    block_type=block["block_type"],
                    content=block.get("content", ""),
                    links=[LinkRef(**link) for link in block.get("links", [])],
                    language=block.get("language", ""),
                    admonition_type=block.get("admonition_type", ""),
                    table_format=block.get("table_format", ""),
                )
                for block in section_data.get("blocks", [])
            ],
        )
        for block in section.blocks:
            for chunk_type, content_text, links in _chunk_block(
                block,
                section,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            ):
                chunks.append(
                    {
                        "chunk_type": chunk_type,
                        "content_text": content_text,
                        "section_path": section.section_path[:512],
                        "heading": section.heading[:256],
                        "heading_level": section.heading_level,
                        "links": [link.to_dict() for link in links],
                        "parser_version": PARSER_VERSION,
                        "chunker_version": CHUNKER_VERSION,
                        "estimated_tokens": estimate_tokens(content_text),
                    }
                )
    return chunks


def build_citation_url(file_path: str, base_url: str) -> str:
    """Build a Kubeflow docs citation URL from a repository path."""
    path_parts = file_path.split("/")
    if "content/en/docs" in file_path:
        docs_index = path_parts.index("docs")
        url_path = "/".join(path_parts[docs_index + 1 :])
        url_path = os.path.splitext(url_path)[0]
        if url_path.endswith("/_index"):
            url_path = url_path[: -len("/_index")]
        citation_url = f"{base_url.rstrip('/')}/{url_path}"
    else:
        citation_url = f"{base_url.rstrip('/')}/{file_path}"
    return citation_url[:1024]


def infer_doc_status(file_path: str) -> str:
    if "components/pipelines/legacy-v1/" in file_path:
        return "deprecated"
    return "active"


def infer_doc_type(file_path: str, frontmatter: dict[str, Any]) -> str:
    if frontmatter.get("manualLink"):
        return "redirect"
    if "/releases/kubeflow-" in file_path.replace("\\", "/"):
        return "release"
    if file_path.endswith("_index.md"):
        return "nav"
    return "documentation"


def _normalize_iso_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    if not text or not ISO_DATE_RE.match(text):
        return None
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None
    return text


def _iso_date_to_epoch(iso_date: str) -> int:
    dt = datetime.strptime(iso_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def extract_release_date(
    *,
    doc_type: str,
    source_text: str,
    frontmatter: dict[str, Any],
) -> int | None:
    """Return UTC epoch seconds for release GA date, or None when unknown."""
    if doc_type != "release":
        return None

    candidates: list[str] = []
    for key in FRONTMATTER_RELEASE_DATE_KEYS:
        iso_date = _normalize_iso_date(frontmatter.get(key))
        if iso_date:
            candidates.append(iso_date)

    html_match = RELEASE_DATE_HTML_RE.search(source_text)
    if html_match:
        candidates.append(html_match.group(1))

    gfm_match = RELEASE_DATE_GFM_RE.search(source_text)
    if gfm_match:
        candidates.append(gfm_match.group(1))

    for iso_date in candidates:
        normalized = _normalize_iso_date(iso_date)
        if normalized and normalized in source_text:
            return _iso_date_to_epoch(normalized)
    return None


def build_milvus_records(
    file_data: dict[str, Any],
    *,
    repo_name: str,
    base_url: str,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    """Parse and chunk one downloaded file record into Milvus-ready dicts."""
    content = file_data.get("content", "") or ""
    file_path = file_data.get("path", "") or ""
    file_name = file_data.get("file_name") or os.path.basename(file_path)

    parsed = parse_canonical_document(content)
    frontmatter = parsed.frontmatter
    doc_type = infer_doc_type(file_path, frontmatter)
    release_date = extract_release_date(
        doc_type=doc_type,
        source_text=content,
        frontmatter=frontmatter,
    )

    if doc_type == "redirect":
        manual_link = str(frontmatter.get("manualLink", "")).strip()
        chunks = [
            {
                "chunk_type": "redirect",
                "content_text": f"Redirect: {parsed.title}. {manual_link}"[:MAX_CONTENT_TEXT_CHARS],
                "section_path": parsed.title[:512],
                "heading": parsed.title[:256],
                "heading_level": 0,
                "links": [{"text": parsed.title, "url": manual_link}] if manual_link else [],
                "parser_version": PARSER_VERSION,
                "chunker_version": CHUNKER_VERSION,
                "estimated_tokens": estimate_tokens(manual_link),
            }
        ]
    elif doc_type == "nav" and not any(section.blocks for section in parsed.sections):
        nav_text = f"Section: {parsed.title}. {parsed.description}".strip()
        chunks = [
            {
                "chunk_type": "nav",
                "content_text": nav_text[:MAX_CONTENT_TEXT_CHARS],
                "section_path": parsed.title[:512],
                "heading": parsed.title[:256],
                "heading_level": 0,
                "links": [],
                "parser_version": PARSER_VERSION,
                "chunker_version": CHUNKER_VERSION,
                "estimated_tokens": estimate_tokens(nav_text),
            }
        ]
    else:
        chunks = chunk_canonical_document(
            parsed,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )

    citation_url = build_citation_url(file_path, base_url)
    if doc_type == "redirect" and frontmatter.get("manualLink"):
        citation_url = str(frontmatter["manualLink"])[:1024]

    file_unique_id = f"{repo_name}:{file_path}"
    doc_status = infer_doc_status(file_path)
    records: list[dict[str, Any]] = []

    for chunk_idx, chunk in enumerate(chunks):
        records.append(
            {
                "file_unique_id": file_unique_id,
                "repo_name": repo_name,
                "file_path": file_path,
                "file_name": file_name,
                "citation_url": citation_url,
                "chunk_index": chunk_idx,
                "content_text": chunk["content_text"][:MAX_CONTENT_TEXT_CHARS],
                "title": parsed.title[:256],
                "weight": parsed.weight,
                "doc_type": doc_type[:32],
                "version": str(frontmatter.get("version") or "")[:32],
                "release_date": release_date,
                "chunk_type": chunk["chunk_type"][:32],
                "section_path": chunk.get("section_path", "")[:512],
                "heading": chunk.get("heading", "")[:256],
                "doc_status": doc_status[:32],
                "parser_version": chunk.get("parser_version", PARSER_VERSION),
                "chunker_version": chunk.get("chunker_version", CHUNKER_VERSION),
                "heading_level": int(chunk.get("heading_level", 0)),
                "estimated_tokens": int(chunk.get("estimated_tokens", 0)),
                "links": chunk.get("links", []),
            }
        )

    return records


def parse_and_chunk_file(
    file_data: dict[str, Any],
    *,
    repo_name: str,
    base_url: str,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> dict[str, Any]:
    """Return canonical parse tree and Milvus-ready chunk records for one file."""
    parsed = parse_canonical_document(file_data.get("content", "") or "")
    records = build_milvus_records(
        file_data,
        repo_name=repo_name,
        base_url=base_url,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )
    return {
        "canonical": parsed.to_dict(),
        "chunks": records,
        "parser_version": PARSER_VERSION,
        "chunker_version": CHUNKER_VERSION,
    }


def dumps_jsonl(records: Iterable[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)


__all__ = [
    "PARSER_VERSION",
    "CHUNKER_VERSION",
    "DEFAULT_TARGET_TOKENS",
    "DEFAULT_OVERLAP_TOKENS",
    "CanonicalBlock",
    "CanonicalDocument",
    "CanonicalSection",
    "build_citation_url",
    "build_milvus_records",
    "chunk_canonical_document",
    "extract_release_date",
    "dumps_jsonl",
    "estimate_tokens",
    "parse_and_chunk_file",
    "parse_canonical_document",
    "split_prose_by_tokens",
]
