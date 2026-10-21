"""
Code Ingestion — Chunker Component

Post-processes AST parser output to enforce token limits and add
context headers to each chunk.

Features:
  - Enforces 50-512 token limits
  - Prepends context header: # File: ... | Symbol: ... | Lang: ...
  - Splits oversized chunks at logical boundaries (blank lines)
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Try to use tiktoken
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_ENCODER.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)

MIN_TOKENS = 50
MAX_TOKENS = 512


def build_path_hints(chunk: Dict[str, Any]) -> str:
    """Build a normalized path-hint string for retrieval context."""
    raw = " ".join(
        str(chunk.get(key, ""))
        for key in ("file_path", "folder_context", "symbol_name")
    )
    expanded = raw.replace("/", " ").replace("_", " ").replace("-", " ")
    expanded = "".join(
        (
            f" {char}" if index > 0 and char.isupper() and expanded[index - 1].islower() else char
        )
        for index, char in enumerate(expanded)
    )
    normalized = " ".join(expanded.split())
    return normalized.lower()


def make_context_header(chunk: Dict[str, Any]) -> str:
    """Create a context header string for a code chunk.

    This header is prepended to the chunk text before embedding to help
    the model understand the code's origin and purpose.

    Args:
        chunk: Chunk dict with file_path, symbol_name, language.

    Returns:
        Context header string.
    """
    lines = [
        (
            f"# File: {chunk.get('file_path', 'unknown')} "
            f"| Symbol: {chunk.get('symbol_name', 'unknown')} "
            f"| Lang: {chunk.get('language', 'unknown')} "
            f"| Folder: {chunk.get('folder_context', 'unknown')}"
        )
    ]
    path_hints = build_path_hints(chunk)
    if path_hints:
        lines.append(f"# Path Hints: {path_hints}")
    return "\n".join(lines)


def split_oversized_chunk(text: str, max_tokens: int) -> List[str]:
    """Split an oversized chunk at logical boundaries.

    Tries to split at blank lines first, then single newlines,
    then falls back to word splitting.

    Args:
        text: Text to split.
        max_tokens: Maximum tokens per sub-chunk.

    Returns:
        List of sub-chunk strings.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    # Try blank line split first
    for sep in ["\n\n", "\n"]:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        chunks = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if count_tokens(candidate) > max_tokens:
                if current.strip():
                    chunks.append(current.strip())
                current = part
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())

        if len(chunks) > 1:
            return chunks

    # Last resort: word split
    words = text.split()
    chunks = []
    current_words = []

    for word in words:
        current_words.append(word)
        if count_tokens(" ".join(current_words)) > max_tokens:
            if len(current_words) > 1:
                chunks.append(" ".join(current_words[:-1]))
                current_words = [word]

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def process_chunks(
    raw_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Post-process AST parser chunks with token limits and context headers.

    Args:
        raw_chunks: List of raw chunk dicts from AST parser.

    Returns:
        List of processed chunk dicts ready for embedding.
    """
    processed = []
    skipped_short = 0
    split_count = 0
    # Track index per file_path
    file_indices = {}

    for chunk in raw_chunks:
        # Add context header
        header = make_context_header(chunk)
        full_text = f"{header}\n\n{chunk['chunk_text']}"
        token_count = count_tokens(full_text)

        if token_count < MIN_TOKENS:
            skipped_short += 1
            continue

        if token_count <= MAX_TOKENS:
            processed_chunk = chunk.copy()
            fp = chunk.get("file_path", "unknown")
            ci = file_indices.get(fp, 0)
            processed_chunk["chunk_text"] = full_text[:8192]
            processed_chunk["token_count"] = token_count
            processed_chunk["chunk_index"] = ci
            processed.append(processed_chunk)
            file_indices[fp] = ci + 1
        else:
            # Split oversized chunk
            sub_chunks = split_oversized_chunk(full_text, MAX_TOKENS)
            split_count += 1

            for idx, sub_text in enumerate(sub_chunks):
                sub_tokens = count_tokens(sub_text)
                if sub_tokens < MIN_TOKENS:
                    continue

                sub_chunk = chunk.copy()
                fp = chunk.get("file_path", "unknown")
                ci = file_indices.get(fp, 0)
                sub_chunk["chunk_id"] = hashlib.sha256(
                    f"{chunk['chunk_id']}::{idx}".encode()
                ).hexdigest()[:32]
                sub_chunk["chunk_text"] = sub_text[:8192]
                sub_chunk["token_count"] = sub_tokens
                sub_chunk["chunk_index"] = ci
                processed.append(sub_chunk)
                file_indices[fp] = ci + 1

    logger.info(
        "Chunker: %d input -> %d output (%d short skipped, %d split)",
        len(raw_chunks), len(processed), skipped_short, split_count,
    )
    return processed


def save_chunks(chunks: List[Dict[str, Any]], output_path: str) -> None:
    """Save chunks to a JSONL file.

    Args:
        chunks: List of chunk dicts.
        output_path: Path to write the file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info("Saved %d chunks to %s", len(chunks), output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_chunks = [
        {
            "chunk_id": "abc123",
            "file_path": "apps/pipeline/upstream/kfp/v2/compiler.py",
            "extension": ".py",
            "language": "python",
            "symbol_name": "function:compile_pipeline",
            "chunk_text": "def compile_pipeline(pipeline_func):\n    \"\"\"Compile a pipeline function.\"\"\"\n    return compiled",
            "start_line": 10,
            "end_line": 12,
            "commit_sha": "deadbeef",
            "folder_context": "apps",
        },
    ]

    result = process_chunks(test_chunks)
    logger.info("=== Code Chunker Test ===")
    for c in result:
        logger.info("  %s (tokens=%d)", c["symbol_name"], c.get("token_count", 0))
        logger.info("  Text preview: %s...", c["chunk_text"][:100])
