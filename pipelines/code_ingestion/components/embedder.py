"""
Code Ingestion — Embedder Component

Embeds code chunks using configurable embedding model.
Identical to docs embedder but imports from shared utilities.

The context header prepended by the chunker is included in the
embedding input so vectors capture both code semantics and file location.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from pipelines.shared.embedding_utils import EmbeddingClient

logger = logging.getLogger(__name__)


def embed_code_chunks(
    chunks: List[Dict[str, Any]],
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Embed all code chunks and add embeddings to each chunk dict.

    Args:
        chunks: List of chunk dicts (must have 'chunk_text' key).
        batch_size: Batch size for embedding.

    Returns:
        Same chunk dicts with added 'embedding' key.
    """
    if not chunks:
        logger.warning("No chunks to embed.")
        return []

    client = EmbeddingClient(batch_size=batch_size)
    texts = [chunk["chunk_text"] for chunk in chunks]

    logger.info("Embedding %d code chunks with model: %s", len(texts), client.model_name)
    embeddings = client.embed_texts(texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    logger.info("Embedding complete. %d code chunks embedded.", len(chunks))
    return chunks


def load_chunks(input_path: str) -> List[Dict[str, Any]]:
    """Load chunks from a JSONL file."""
    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def save_embedded_chunks(chunks: List[Dict[str, Any]], output_path: str) -> None:
    """Save embedded chunks to a JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info("Saved %d embedded chunks to %s", len(chunks), output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    test_chunks = [
        {
            "chunk_id": "test001",
            "chunk_text": "# File: apps/kfp/compiler.py | Symbol: func:compile | Lang: python\n\ndef compile(): pass",
            "file_path": "apps/kfp/compiler.py",
            "language": "python",
        },
    ]
    result = embed_code_chunks(test_chunks)
    for c in result:
        logger.info("  chunk_id=%s dim=%d", c["chunk_id"], len(c.get("embedding", [])))
