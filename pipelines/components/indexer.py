from kfp import dsl
from kfp.dsl import Input, Dataset

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pymilvus", "numpy"]
)
def milvus_indexer_component(
    processed_dataset: Input[Dataset],
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
    vector_dim: int = 768
):
    from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
    import json
    from datetime import datetime

    connections.connect("default", host=milvus_host, port=milvus_port)

    # Schema setup
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="file_unique_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="citation_url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        FieldSchema(name="last_updated", dtype=DataType.INT64)
    ]

    schema = CollectionSchema(fields, "RAG collection for documentation")
    
    if not utility.has_collection(collection_name):
        collection = Collection(collection_name, schema)
        print(f"Created collection: {collection_name}")
    else:
        collection = Collection(collection_name)
        print(f"Using existing collection: {collection_name}")

    timestamp = int(datetime.now().timestamp())
    records = []

    with open(processed_dataset.path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            records.append({
                "file_unique_id": record["file_unique_id"],
                "repo_name": record["repo_name"],
                "file_path": record["file_path"],
                "citation_url": record["citation_url"],
                "chunk_index": record["chunk_index"],
                "content_text": record["content_text"][:2000],
                "vector": record["embedding"],
                "last_updated": timestamp
            })

    if records:
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            collection.insert(batch)
        
        collection.flush()
        
        # Create index if it doesn't exist
        if not collection.has_index():
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index("vector", index_params)
        
        collection.load()
        print(f"✅ Indexed {len(records)} records.")
