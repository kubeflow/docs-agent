from feast import Entity, FeatureView, Field, FileSource
from feast.types import String, Int64, Float32, Array, UnixTimestamp
from datetime import timedelta

doc_chunk = Entity(name="doc_chunk", join_keys=["file_unique_id"])

docs_source = FileSource(
    path="data/embedded_docs.parquet",
    timestamp_field="event_timestamp",
)

docs_embeddings = FeatureView(
    name="docs_rag",
    entities=[doc_chunk],
    schema=[
        Field(name="file_unique_id", dtype=String),
        Field(name="repo_name", dtype=String),
        Field(name="file_path", dtype=String),
        Field(name="file_name", dtype=String),
        Field(name="citation_url", dtype=String),
        Field(name="chunk_index", dtype=Int64),
        Field(name="content_text", dtype=String),
        Field(
            name="vector",
            dtype=Array(Float32),
            vector_index=True,
            vector_search_metric="COSINE",
        ),
        Field(name="event_timestamp", dtype=UnixTimestamp),
    ],
    source=docs_source,
    ttl=timedelta(days=30),
)
