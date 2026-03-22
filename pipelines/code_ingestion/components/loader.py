"""
Code Ingestion — Loader Component

Loads embedded code chunks into the Milvus code_collection.
Uses upsert pattern with chunk_id as primary key.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)

from pipelines.shared.milvus_utils import connect, create_collection_if_not_exists, upsert_batch
from backend.schemas.code_collection_schema import (
    COLLECTION_NAME,
    get_code_fields,
    get_code_index_params,
)

logger = logging.getLogger(__name__)


def should_recreate_collection() -> bool:
    """Return whether the loader should drop and recreate the collection.

    This is disabled by default so local re-runs preserve previously indexed
    data and rely on primary-key upserts instead of destructive reloads.
    """
    return os.environ.get("MILVUS_DROP_EXISTING", "false").lower() == "true"


def load_to_milvus(
    chunks: List[Dict[str, Any]],
    collection_name: str = None,
) -> Dict[str, int]:
    """Load embedded code chunks into Milvus code_collection.

    Args:
        chunks: List of chunk dicts with embeddings.
        collection_name: Override collection name.

    Returns:
        Ingestion summary with inserted, failed, total counts.
    """
    col_name = collection_name or COLLECTION_NAME

    connect()

    # Recreate only when explicitly requested.
    if should_recreate_collection() and utility.has_collection(col_name):
        utility.drop_collection(col_name)
        logger.info("Dropped existing collection %s for schema refresh", col_name)

    fields = get_code_fields()
    index_params = get_code_index_params()
    collection = create_collection_if_not_exists(
        collection_name=col_name,
        fields=fields,
        description="Kubeflow manifests code chunks for RAG retrieval",
        index_field="embedding",
        index_params=index_params,
    )

    rows = []
    for chunk in chunks:
        if "embedding" not in chunk:
            continue

        row = {
            "chunk_id": str(chunk["chunk_id"])[:128],
            "file_path": str(chunk.get("file_path", ""))[:512],
            "extension": str(chunk.get("extension", ""))[:16],
            "language": str(chunk.get("language", ""))[:32],
            "symbol_name": str(chunk.get("symbol_name", ""))[:256],
            "folder_context": str(chunk.get("folder_context", ""))[:128],
            "chunk_text": str(chunk.get("chunk_text", ""))[:8192],
            "start_line": int(chunk.get("start_line", 0)),
            "end_line": int(chunk.get("end_line", 0)),
            "commit_sha": str(chunk.get("commit_sha", ""))[:64],
            "chunk_index": int(chunk.get("chunk_index", 0)),
            "embedding": chunk["embedding"],
        }
        rows.append(row)

    if not rows:
        return {"inserted": 0, "failed": 0, "total": 0, "skipped": len(chunks)}

    summary = upsert_batch(collection, rows, batch_size=100)
    summary["skipped"] = len(chunks) - len(rows)

    logger.info(
        "Code ingestion: %d inserted, %d failed, %d skipped",
        summary["inserted"], summary["failed"], summary["skipped"],
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("=== Code Loader Smoke Test ===")
    logger.info("Requires Milvus at localhost:19530")
