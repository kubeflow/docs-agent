# server/vector_services/milvus_client.py

from functools import lru_cache
import os
from typing import List, Dict, Any

from pymilvus import (
    connections,
    Collection,
    utility,
)


MILVUS_HOST_ENV = "MILVUS_HOST"
MILVUS_PORT_ENV = "MILVUS_PORT"
MILVUS_COLLECTION_ENV = "MILVUS_COLLECTION"

DEFAULT_MILVUS_HOST = "milvus"
DEFAULT_MILVUS_PORT = "19530"
DEFAULT_MILVUS_COLLECTION = "docs_rag"


@lru_cache(maxsize=1)
def _connect_default() -> None:
    """
    Establish a shared connection to Milvus.

    This is called implicitly by get_milvus_collection.
    """
    host = os.getenv(MILVUS_HOST_ENV, DEFAULT_MILVUS_HOST)
    port = os.getenv(MILVUS_PORT_ENV, DEFAULT_MILVUS_PORT)

    connections.connect(
        alias="default",
        host=host,
        port=port,
    )


@lru_cache(maxsize=1)
def get_milvus_collection() -> Collection:
    """
    Return a shared Collection handle.

    Lazily connects to Milvus and opens the configured collection.
    """
    _connect_default()
    collection_name = os.getenv(MILVUS_COLLECTION_ENV, DEFAULT_MILVUS_COLLECTION)

    if not utility.has_collection(collection_name):
        raise RuntimeError(f"Milvus collection '{collection_name}' does not exist")

    return Collection(collection_name)


def search_vectors(
    query_vectors: List[List[float]],
    top_k: int = 5,
    search_params: Dict[str, Any] | None = None,
    output_fields: List[str] | None = None,
):
    """
    Convenience wrapper around Milvus collection.search.
    """
    collection = get_milvus_collection()

    if search_params is None:
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10},
        }

    if output_fields is None:
        output_fields = []

    results = collection.search(
        data=query_vectors,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=output_fields,
    )

    return results