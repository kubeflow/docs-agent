"""
Milvus schema definition for the code_collection.

Stores chunked and embedded code from kubeflow/manifests repository.
Supports Python, Go, YAML, and Markdown file types.
Uses HNSW index with COSINE metric for fast ANN retrieval.

Dimension defaults to 384 (all-MiniLM-L6-v2). Override via EMBEDDING_MODEL env var.
"""

import logging
import os
import sys

from pymilvus import CollectionSchema, DataType, FieldSchema

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pipelines.shared.embedding_utils import get_embedding_dimension

logger = logging.getLogger(__name__)

COLLECTION_NAME = "code_collection"


def get_code_fields(dim: int = None) -> list:
    """Define the field schema for code_collection.

    Args:
        dim: Embedding vector dimension. Auto-detected from EMBEDDING_MODEL if None.

    Returns:
        List of FieldSchema objects.
    """
    if dim is None:
        dim = get_embedding_dimension()

    fields = [
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=128,
            is_primary=True,
            description="Unique chunk identifier (hash of file_path + symbol + index)",
        ),
        FieldSchema(
            name="file_path",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Relative file path within the repository",
        ),
        FieldSchema(
            name="extension",
            dtype=DataType.VARCHAR,
            max_length=16,
            description="File extension (e.g., .py, .go, .yaml)",
        ),
        FieldSchema(
            name="language",
            dtype=DataType.VARCHAR,
            max_length=32,
            description="Programming language (python, go, yaml, markdown)",
        ),
        FieldSchema(
            name="symbol_name",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Function/class/struct/resource name",
        ),
        FieldSchema(
            name="folder_context",
            dtype=DataType.VARCHAR,
            max_length=128,
            description="Top-level folder for domain context (e.g., apps, common)",
        ),
        FieldSchema(
            name="chunk_text",
            dtype=DataType.VARCHAR,
            max_length=8192,
            description="The actual code/content chunk text",
        ),
        FieldSchema(
            name="start_line",
            dtype=DataType.INT64,
            description="Starting line number in the source file",
        ),
        FieldSchema(
            name="end_line",
            dtype=DataType.INT64,
            description="Ending line number in the source file",
        ),
        FieldSchema(
            name="commit_sha",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Git commit SHA for provenance tracking",
        ),
        FieldSchema(
            name="chunk_index",
            dtype=DataType.INT64,
            description="Index of this chunk within the file (for compatibility)",
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
            description=f"Dense embedding vector ({dim} dimensions)",
        ),
    ]
    return fields


def get_code_schema(dim: int = None) -> CollectionSchema:
    """Create the full CollectionSchema for code_collection.

    Args:
        dim: Embedding vector dimension.

    Returns:
        CollectionSchema object.
    """
    fields = get_code_fields(dim)
    schema = CollectionSchema(
        fields=fields,
        description="Kubeflow manifests code chunks with embeddings for RAG retrieval",
    )
    return schema


def get_code_index_params() -> dict:
    """Get the HNSW index parameters for the embedding field.

    Returns:
        Dict of index parameters for Milvus.
    """
    return {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    schema = get_code_schema()
    logger.info("code_collection schema:")
    for field in schema.fields:
        logger.info(
            "  %s: %s (max_length=%s, dim=%s, primary=%s)",
            field.name,
            field.dtype.name,
            getattr(field, "max_length", "-"),
            getattr(field, "dim", "-"),
            field.is_primary,
        )
    logger.info("Index params: %s", get_code_index_params())
