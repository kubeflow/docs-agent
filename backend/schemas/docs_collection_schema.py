"""
Milvus schema definition for the docs_collection.

Stores chunked and embedded Kubeflow documentation from kubeflow.org.
Uses HNSW index with COSINE metric for fast ANN retrieval.

Dimension defaults to 384 (all-MiniLM-L6-v2). Override via EMBEDDING_MODEL env var.
"""

import logging
import os
import sys

from pymilvus import CollectionSchema, DataType, FieldSchema

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pipelines.shared.embedding_utils import get_embedding_dimension

logger = logging.getLogger(__name__)

COLLECTION_NAME = "docs_collection"


def get_docs_fields(dim: int = None) -> list:
    """Define the field schema for docs_collection.

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
            description="Unique chunk identifier (hash of url + chunk_index)",
        ),
        FieldSchema(
            name="source_url",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Original page URL from kubeflow.org",
        ),
        FieldSchema(
            name="page_title",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Page title extracted from content",
        ),
        FieldSchema(
            name="heading",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="H2/H3 heading this chunk belongs to",
        ),
        FieldSchema(
            name="section",
            dtype=DataType.VARCHAR,
            max_length=128,
            description="Top-level docs section (e.g., components, started)",
        ),
        FieldSchema(
            name="chunk_text",
            dtype=DataType.VARCHAR,
            max_length=16384,
            description="The actual chunk text content",
        ),
        FieldSchema(
            name="token_count",
            dtype=DataType.INT64,
            description="Number of tokens in this chunk",
        ),
        FieldSchema(
            name="chunk_index",
            dtype=DataType.INT64,
            description="Sequential index of this chunk within its page",
        ),
        FieldSchema(
            name="crawled_at",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="ISO timestamp when the page was crawled",
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
            description=f"Dense embedding vector ({dim} dimensions)",
        ),
    ]
    return fields


def get_docs_schema(dim: int = None) -> CollectionSchema:
    """Create the full CollectionSchema for docs_collection.

    Args:
        dim: Embedding vector dimension.

    Returns:
        CollectionSchema object.
    """
    fields = get_docs_fields(dim)
    schema = CollectionSchema(
        fields=fields,
        description="Kubeflow documentation chunks with embeddings for RAG retrieval",
    )
    return schema


def get_docs_index_params() -> dict:
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
    schema = get_docs_schema()
    logger.info("docs_collection schema:")
    for field in schema.fields:
        logger.info(
            "  %s: %s (max_length=%s, dim=%s, primary=%s)",
            field.name,
            field.dtype.name,
            getattr(field, "max_length", "-"),
            getattr(field, "dim", "-"),
            field.is_primary,
        )
    logger.info("Index params: %s", get_docs_index_params())
