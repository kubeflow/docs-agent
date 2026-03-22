"""
Shared Milvus utilities for docs-agent ingestion pipelines.

Provides connection management, collection creation, upsert, and search
operations with retry logic and exponential backoff.

Configure via environment variables:
  MILVUS_HOST: Milvus server host (default: localhost)
  MILVUS_PORT: Milvus server port (default: 19530)
  MILVUS_TOKEN: Authentication token (default: empty, no auth)
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

logger = logging.getLogger(__name__)


def get_milvus_config() -> Dict[str, str]:
    """Get Milvus connection configuration from environment.

    Returns:
        Dict with host, port, and token.
    """
    return {
        "host": os.environ.get("MILVUS_HOST", "localhost"),
        "port": os.environ.get("MILVUS_PORT", "19530"),
        "token": os.environ.get("MILVUS_TOKEN", ""),
    }


def connect(
    alias: str = "default",
    host: Optional[str] = None,
    port: Optional[str] = None,
    token: Optional[str] = None,
    max_retries: int = 3,
) -> None:
    """Connect to Milvus with retry logic.

    Args:
        alias: Connection alias.
        host: Override for MILVUS_HOST env var.
        port: Override for MILVUS_PORT env var.
        token: Override for MILVUS_TOKEN env var.
        max_retries: Maximum retry attempts.

    Raises:
        ConnectionError: If all retries are exhausted.
    """
    config = get_milvus_config()
    host = host or config["host"]
    port = port or config["port"]
    token = token or config["token"]

    for attempt in range(max_retries):
        try:
            connect_params = {"alias": alias, "host": host, "port": port}
            if token:
                connect_params["token"] = token

            connections.connect(**connect_params)
            logger.info("Connected to Milvus at %s:%s", host, port)
            return
        except Exception as e:
            wait_time = (2 ** attempt) + 1
            logger.warning(
                "Milvus connection failed (attempt %d/%d): %s. Retrying in %ds...",
                attempt + 1,
                max_retries,
                str(e),
                wait_time,
            )
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                raise ConnectionError(
                    f"Failed to connect to Milvus after {max_retries} attempts: {e}"
                ) from e


def create_collection_if_not_exists(
    collection_name: str,
    fields: List[FieldSchema],
    description: str = "",
    index_field: str = "embedding",
    index_params: Optional[Dict[str, Any]] = None,
) -> Collection:
    """Create a Milvus collection if it doesn't already exist.

    Args:
        collection_name: Name of the collection.
        fields: List of FieldSchema objects defining the schema.
        description: Collection description.
        index_field: Name of the vector field to index.
        index_params: Custom index parameters. Defaults to HNSW + COSINE.

    Returns:
        The Milvus Collection object.
    """
    if utility.has_collection(collection_name):
        logger.info("Collection '%s' already exists. Loading.", collection_name)
        collection = Collection(collection_name)
        collection.load()
        return collection

    schema = CollectionSchema(fields, description=description)
    collection = Collection(name=collection_name, schema=schema)
    logger.info("Created collection: %s", collection_name)

    # Default HNSW index params
    if index_params is None:
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        }

    collection.create_index(field_name=index_field, index_params=index_params)
    logger.info(
        "Created HNSW index on '%s' for collection '%s'",
        index_field,
        collection_name,
    )

    collection.load()
    return collection


def upsert_batch(
    collection: Collection,
    rows: List[Dict[str, Any]],
    batch_size: int = 100,
    max_retries: int = 3,
) -> Dict[str, int]:
    """Upsert rows into a Milvus collection in batches.

    Uses the primary key to handle duplicates (Milvus upsert semantics).

    Args:
        collection: The target Milvus collection.
        rows: List of row dicts matching the collection schema.
        batch_size: Number of rows per insert batch.
        max_retries: Retry attempts per batch.

    Returns:
        Dict with counts: inserted, failed, total.
    """
    total = len(rows)
    inserted = 0
    failed = 0

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        success = False
        for attempt in range(max_retries):
            try:
                collection.upsert(batch)
                inserted += len(batch)
                success = True
                logger.info(
                    "Upserted batch %d/%d (%d rows)",
                    batch_num,
                    total_batches,
                    len(batch),
                )
                break
            except Exception as e:
                wait_time = (2 ** attempt) + 1
                logger.warning(
                    "Upsert failed (batch %d, attempt %d/%d): %s. "
                    "Retrying in %ds...",
                    batch_num,
                    attempt + 1,
                    max_retries,
                    str(e),
                    wait_time,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)

        if not success:
            failed += len(batch)
            logger.error("Batch %d permanently failed after %d retries.", batch_num, max_retries)

    collection.flush()
    summary = {"inserted": inserted, "failed": failed, "total": total}
    logger.info("Upsert complete: %s", summary)
    return summary


def search(
    collection: Collection,
    query_vector: List[float],
    top_k: int = 3,
    output_fields: Optional[List[str]] = None,
    search_params: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """Search a Milvus collection by vector similarity.

    Args:
        collection: The Milvus collection to search.
        query_vector: The query embedding vector.
        top_k: Number of results to return.
        output_fields: Fields to include in results.
        search_params: Custom search parameters.
        max_retries: Retry attempts.

    Returns:
        List of result dicts with fields and distance score.
    """
    if search_params is None:
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

    if output_fields is None:
        output_fields = ["chunk_text"]

    for attempt in range(max_retries):
        try:
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=output_fields,
            )

            hits = []
            for hit in results[0]:
                hit_dict = {"id": hit.id, "distance": hit.distance}
                for field in output_fields:
                    hit_dict[field] = hit.entity.get(field)
                hits.append(hit_dict)

            logger.info("Search returned %d results.", len(hits))
            return hits

        except Exception as e:
            wait_time = (2 ** attempt) + 1
            logger.warning(
                "Search failed (attempt %d/%d): %s. Retrying in %ds...",
                attempt + 1,
                max_retries,
                str(e),
                wait_time,
            )
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                logger.error("Search failed after %d retries: %s", max_retries, e)
                return []

    return []


def drop_collection(collection_name: str) -> bool:
    """Drop a collection if it exists.

    Args:
        collection_name: Name of the collection to drop.

    Returns:
        True if dropped, False if it didn't exist.
    """
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        logger.info("Dropped collection: %s", collection_name)
        return True
    logger.info("Collection '%s' does not exist. Nothing to drop.", collection_name)
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick connection test
    try:
        connect()
        logger.info("Milvus connection test: SUCCESS")
        collections = utility.list_collections()
        logger.info("Existing collections: %s", collections)
    except ConnectionError as e:
        logger.error("Milvus connection test: FAILED — %s", e)
