from pymilvus import connections, utility, Collection
from pymilvus import FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer

print("Step 1 - Connecting to Milvus...")
connections.connect("default", host="localhost", port="19530")
print("Connected OK")

print("\nStep 2 - Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
print("Model loaded")

print("\nStep 3 - Creating docs_index collection...")
if utility.has_collection("docs_index"):
    utility.drop_collection("docs_index")

fields = [
    FieldSchema(name="id",           dtype=DataType.INT64,        is_primary=True, auto_id=True),
    FieldSchema(name="content_text", dtype=DataType.VARCHAR,       max_length=2000),
    FieldSchema(name="source_url",   dtype=DataType.VARCHAR,       max_length=500),
    FieldSchema(name="h1",           dtype=DataType.VARCHAR,       max_length=300),
    FieldSchema(name="h2",           dtype=DataType.VARCHAR,       max_length=300),
    FieldSchema(name="h3",           dtype=DataType.VARCHAR,       max_length=300),
    FieldSchema(name="vector",       dtype=DataType.FLOAT_VECTOR,  dim=768),
]
schema     = CollectionSchema(fields, "Kubeflow docs index")
collection = Collection("docs_index", schema)
print("Collection created")

print("\nStep 4 - Inserting sample docs...")
sample_docs = [
    {
        "text": "Kubeflow Pipelines is a platform for building and deploying ML workflows.",
        "url":  "https://kubeflow.org/docs/pipelines",
        "h1":   "Kubeflow Pipelines",
        "h2":   "Overview",
        "h3":   ""
    },
    {
        "text": "KServe provides serverless inferencing on Kubernetes using InferenceService CRD.",
        "url":  "https://kubeflow.org/docs/kserve",
        "h1":   "KServe",
        "h2":   "Installation",
        "h3":   ""
    },
    {
        "text": "To install Kubeflow you need a Kubernetes cluster version 1.20 or higher.",
        "url":  "https://kubeflow.org/docs/started",
        "h1":   "Getting Started",
        "h2":   "Prerequisites",
        "h3":   "Kubernetes version"
    },
]

texts      = [d["text"] for d in sample_docs]
embeddings = model.encode(texts).tolist()

collection.insert([{
    "content_text": d["text"],
    "source_url":   d["url"],
    "h1":           d["h1"],
    "h2":           d["h2"],
    "h3":           d["h3"],
    "vector":       e
} for d, e in zip(sample_docs, embeddings)])

collection.flush()

index_params = {
    "metric_type": "COSINE",
    "index_type":  "IVF_FLAT",
    "params":      {"nlist": 128}
}
collection.create_index("vector", index_params)
collection.load()
print(f"Inserted {collection.num_entities} documents")

print("\nStep 5 - Searching...")
query     = "how to install kubeflow"
query_vec = model.encode(query).tolist()

results = collection.search(
    data=[query_vec],
    anns_field="vector",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["content_text", "source_url", "h1", "h2"]
)

print(f"Query: '{query}'")
print("Results:")
for i, hit in enumerate(results[0]):
    print(f"\n  Result {i+1}:")
    print(f"  score : {hit.score:.4f}")
    print(f"  h1    : {hit.entity.get('h1')}")
    print(f"  h2    : {hit.entity.get('h2')}")
    print(f"  text  : {hit.entity.get('content_text')[:80]}")
    print(f"  url   : {hit.entity.get('source_url')}")

print("\nPhase 1 end to end test PASSED")