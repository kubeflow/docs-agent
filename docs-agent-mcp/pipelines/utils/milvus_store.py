"""Schema, safety gates, and store helpers for the docs ingest pipeline.

Chunk/store KFP steps import this module from the ingest image
(see Dockerfile.pipeline). Tests import it the same way.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType, MilvusClient

from canonical_rag_ingest import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    build_milvus_records,
)
from utils import DEFAULT_EMBEDDING_BATCH_SIZE, DOCS_COLLECTION, embed_texts, truncate_for_tei

SCHEMA_VERSION = 4
SCHEMA_DESCRIPTION = (
    f"RAG lean hybrid collection for documentation (v={SCHEMA_VERSION}, hybrid=bm25+dense)"
)

APPROVED_DOCS_COLLECTION = DOCS_COLLECTION
CLEAN_REBUILD_CONFIRMATION = f"DROP {APPROVED_DOCS_COLLECTION}"
MAINTENANCE_LOCK_ENV = "MILVUS_MAINTENANCE_LOCK"

DENSE_FIELD = "vector"
SPARSE_FIELD = "sparse_vector"
BM25_INPUT_FIELD = "content_text"
DOCUMENT_ID_FIELD = "document_id"
DENSE_DIM = 768
MAX_CONTENT_TEXT_CHARS = 2000
INSERT_BATCH_SIZE = 1000
DELETE_BATCH_SIZE = 100

ANALYZER_PARAMS = {
    "tokenizer": "standard",
    "filter": ["lowercase"],
}

def truncate_utf8(value: Any, max_bytes: int) -> str:
    """Truncate a string without exceeding Milvus VARCHAR byte limits."""
    text = str(value)
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def validate_production_collection_name(collection_name: str) -> None:
    name = (collection_name or "").strip()
    if name != APPROVED_DOCS_COLLECTION:
        raise ValueError(
            f"Refusing unexpected collection name '{collection_name}'. "
            f"Production docs indexing must target exactly '{APPROVED_DOCS_COLLECTION}'."
        )


def validate_clean_rebuild_gates(
    *,
    clean_rebuild: bool,
    clean_rebuild_confirmation: str,
    maintenance_lock_token: str,
    maintenance_lock_env: str | None = None,
) -> None:
    if not clean_rebuild:
        return

    expected_confirmation = CLEAN_REBUILD_CONFIRMATION
    provided = (clean_rebuild_confirmation or "").strip()
    if provided != expected_confirmation:
        raise ValueError(
            f"clean_rebuild requires typed confirmation '{expected_confirmation}', "
            f"got '{provided or '<empty>'}'."
        )

    env_name = maintenance_lock_env or MAINTENANCE_LOCK_ENV
    expected_lock = (os.environ.get(env_name) or "").strip()
    provided_lock = (maintenance_lock_token or "").strip()
    if not expected_lock:
        raise RuntimeError(
            f"{env_name} must be set in the pipeline environment before clean_rebuild."
        )
    if provided_lock != expected_lock:
        raise RuntimeError(
            f"maintenance_lock_token does not match {env_name}; "
            "refusing destructive rebuild."
        )


def milvus_uri(host: str, port: str) -> str:
    host = (host or "").strip()
    port = (port or "").strip()
    if not host:
        raise ValueError("milvus_host is required")
    if not port:
        raise ValueError("milvus_port is required")
    return f"http://{host}:{port}"


def check_milvus_health(client: MilvusClient) -> None:
    """Refuse indexing when Milvus is unreachable or not ready."""
    try:
        version = client.get_server_version()
    except Exception as exc:  # pragma: no cover - exercised via mocks in tests
        raise RuntimeError(f"Milvus health check failed: {exc}") from exc
    if not version or not str(version).strip():
        raise RuntimeError("Milvus health check failed: empty server version")


def build_lean_v4_schema() -> CollectionSchema:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name=DOCUMENT_ID_FIELD, dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(
            name=BM25_INPUT_FIELD,
            dtype=DataType.VARCHAR,
            max_length=MAX_CONTENT_TEXT_CHARS,
            enable_analyzer=True,
            enable_match=True,
            analyzer_params=ANALYZER_PARAMS,
        ),
        FieldSchema(name=DENSE_FIELD, dtype=DataType.FLOAT_VECTOR, dim=DENSE_DIM),
        FieldSchema(name=SPARSE_FIELD, dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="citation_url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="release_date", dtype=DataType.INT64, nullable=True),
    ]
    schema = CollectionSchema(fields, SCHEMA_DESCRIPTION)
    schema.add_function(
        Function(
            name="content_text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=[BM25_INPUT_FIELD],
            output_field_names=[SPARSE_FIELD],
        )
    )
    return schema


def build_v4_index_params() -> Any:
    params = MilvusClient.prepare_index_params()
    params.add_index(field_name=DENSE_FIELD, index_type="FLAT", metric_type="COSINE")
    params.add_index(
        field_name=SPARSE_FIELD,
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    return params


def schema_version_matches(description: str | None) -> bool:
    return bool(description and f"v={SCHEMA_VERSION}" in description)


def validate_embedding_vectors(vectors: Sequence[Sequence[float]], *, dense_dim: int = DENSE_DIM) -> None:
    for index, vector in enumerate(vectors):
        if not isinstance(vector, Sequence):
            raise ValueError(f"embedding at index {index} is not a sequence")
        if len(vector) != dense_dim:
            raise ValueError(
                f"embedding at index {index} expected {dense_dim}-dim vector, got {len(vector)}"
            )


def chunk_github_jsonl(
    github_data_path: str,
    *,
    repo_name: str,
    base_url: str,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ingest_module: Any | None = None,
) -> list[dict[str, Any]]:
    """Parse/chunk downloaded GitHub JSONL using canonical ingestion logic."""
    build_records = ingest_module.build_milvus_records if ingest_module else build_milvus_records
    records: list[dict[str, Any]] = []

    with open(github_data_path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            file_data = json.loads(stripped)
            file_records = build_records(
                file_data,
                repo_name=repo_name,
                base_url=base_url,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            )
            records.extend(file_records)

    if not records:
        raise ValueError("No chunk records produced from GitHub dataset")
    return records


def embed_chunk_records(
    records: list[dict[str, Any]],
    *,
    embeddings_service_url: str,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Embed every chunk via production TEI and validate 768-d vectors."""
    if not records:
        return records

    batch_size = max(1, int(embedding_batch_size))
    texts = [truncate_for_tei(record["content_text"]) for record in records]
    vectors = embed_texts(
        texts,
        embeddings_service_url,
        batch_size=batch_size,
    )
    validate_embedding_vectors(vectors)

    for record, vector in zip(records, vectors):
        record["embedding"] = vector
    return records


def prepare_v4_insert_row(record: dict[str, Any]) -> dict[str, Any]:
    document_id = record.get("file_unique_id") or record.get(DOCUMENT_ID_FIELD)
    if not document_id:
        raise ValueError("record is missing file_unique_id/document_id")

    row: dict[str, Any] = {
        DOCUMENT_ID_FIELD: truncate_utf8(document_id, 512),
        BM25_INPUT_FIELD: truncate_utf8(record["content_text"], MAX_CONTENT_TEXT_CHARS),
        DENSE_FIELD: record["embedding"],
        "chunk_index": int(record["chunk_index"]),
        "citation_url": truncate_utf8(record["citation_url"], 1024),
        "file_path": truncate_utf8(record["file_path"], 512),
        "title": truncate_utf8(record.get("title", ""), 256),
        "section_path": truncate_utf8(record.get("section_path", ""), 512),
        "doc_type": truncate_utf8(record.get("doc_type", "documentation"), 32),
        "version": truncate_utf8(record.get("version", ""), 32),
    }
    release_date = record.get("release_date")
    if release_date is not None:
        row["release_date"] = int(release_date)
    return row


def compute_validation_metrics(
    *,
    collection_name: str,
    records: Sequence[dict[str, Any]],
    inserted_count: int,
    entity_count: int,
    dense_ready: bool,
    sparse_ready: bool,
    clean_rebuild: bool,
) -> dict[str, Any]:
    release_docs = [record for record in records if record.get("doc_type") == "release"]
    release_with_date = [
        record for record in release_docs if record.get("release_date") is not None
    ]
    release_doc_count = len(release_docs)
    release_date_count = len(release_with_date)
    release_date_fill_rate = (
        round(release_date_count / release_doc_count, 4) if release_doc_count else 0.0
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "schema_description": SCHEMA_DESCRIPTION,
        "collection_name": collection_name,
        "clean_rebuild": clean_rebuild,
        "record_count": len(records),
        "inserted_count": inserted_count,
        "entity_count": entity_count,
        "dense_dim": DENSE_DIM,
        "release_doc_count": release_doc_count,
        "release_date_count": release_date_count,
        "release_date_fill_rate": release_date_fill_rate,
        "dense_index_ready": dense_ready,
        "sparse_index_ready": sparse_ready,
        "bm25_index_ready": sparse_ready,
    }


def store_embedded_records(
    embedded_data_path: str,
    *,
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
    clean_rebuild: bool,
    clean_rebuild_confirmation: str,
    maintenance_lock_token: str,
    client: MilvusClient | None = None,
) -> dict[str, Any]:
    """Read embedded JSONL and write the docs collection."""
    validate_production_collection_name(collection_name)
    validate_clean_rebuild_gates(
        clean_rebuild=clean_rebuild,
        clean_rebuild_confirmation=clean_rebuild_confirmation,
        maintenance_lock_token=maintenance_lock_token,
    )

    milvus_user = os.environ.get("MILVUS_USER", "root")
    milvus_password = os.environ.get("MILVUS_PASSWORD", "")
    if client is None and not milvus_password:
        raise RuntimeError("MILVUS_PASSWORD must be set via pipeline secret (not in source code)")

    milvus = client or MilvusClient(
        uri=milvus_uri(milvus_host, milvus_port),
        user=milvus_user,
        password=milvus_password,
    )
    check_milvus_health(milvus)

    records: list[dict[str, Any]] = []
    with open(embedded_data_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if "embedding" not in record:
                raise ValueError(f"line {line_number}: missing embedding")
            validate_embedding_vectors([record["embedding"]])
            records.append(record)

    if not records:
        raise ValueError("No embedded records found")

    if clean_rebuild and milvus.has_collection(collection_name):
        milvus.drop_collection(collection_name)

    if milvus.has_collection(collection_name):
        info = milvus.describe_collection(collection_name)
        description = info.get("description") or ""
        if not schema_version_matches(description):
            raise RuntimeError(
                f"Collection '{collection_name}' has incompatible schema description "
                f"'{description}'. Expected marker v={SCHEMA_VERSION}. "
                "Use clean_rebuild=true with typed confirmation to recreate."
            )
    else:
        milvus.create_collection(
            collection_name=collection_name,
            schema=build_lean_v4_schema(),
            index_params=build_v4_index_params(),
        )
        print(f"Created new collection: {collection_name} (schema v={SCHEMA_VERSION})")

    rows = [prepare_v4_insert_row(record) for record in records]

    if not clean_rebuild:
        unique_ids = sorted({row[DOCUMENT_ID_FIELD] for row in rows})
        deleted = 0
        for start in range(0, len(unique_ids), DELETE_BATCH_SIZE):
            batch_ids = unique_ids[start : start + DELETE_BATCH_SIZE]
            quoted = ", ".join(f'"{doc_id}"' for doc_id in batch_ids)
            result = milvus.delete(
                collection_name=collection_name,
                filter=f"{DOCUMENT_ID_FIELD} in [{quoted}]",
            )
            deleted += int(result.get("delete_count", 0) or 0)
        if deleted:
            print(f"Deleted {deleted} existing chunks for {len(unique_ids)} documents")

    inserted_count = 0
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= INSERT_BATCH_SIZE:
            milvus.insert(collection_name=collection_name, data=batch)
            inserted_count += len(batch)
            batch.clear()
    if batch:
        milvus.insert(collection_name=collection_name, data=batch)
        inserted_count += len(batch)

    milvus.flush(collection_name)
    milvus.load_collection(collection_name)

    stats = milvus.get_collection_stats(collection_name)
    entity_count = int(stats.get("row_count", 0) or 0)

    indexes = milvus.list_indexes(collection_name=collection_name)
    index_names = set()
    for item in indexes:
        if isinstance(item, str):
            index_names.add(item)
        elif isinstance(item, dict):
            index_names.add(item.get("index_name") or item.get("field_name"))

    metrics = compute_validation_metrics(
        collection_name=collection_name,
        records=records,
        inserted_count=inserted_count,
        entity_count=entity_count,
        dense_ready=DENSE_FIELD in index_names,
        sparse_ready=SPARSE_FIELD in index_names,
        clean_rebuild=clean_rebuild,
    )
    print("VALIDATION_METRICS=" + json.dumps(metrics, ensure_ascii=False, separators=(",", ":")))
    return metrics

