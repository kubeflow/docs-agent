from pymilvus import connections, utility, Collection
from pymilvus import FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer

print("Connecting to Milvus...")
connections.connect("default", host="localhost", port="19530")

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Drop if exists
if utility.has_collection("code_index"):
    utility.drop_collection("code_index")
    print("Dropped existing code_index")

# Create code_index schema
fields = [
    FieldSchema(name="id",           dtype=DataType.INT64,        is_primary=True, auto_id=True),
    FieldSchema(name="content_text", dtype=DataType.VARCHAR,       max_length=4000),
    FieldSchema(name="source_url",   dtype=DataType.VARCHAR,       max_length=500),
    FieldSchema(name="kind",         dtype=DataType.VARCHAR,       max_length=100),
    FieldSchema(name="name",         dtype=DataType.VARCHAR,       max_length=200),
    FieldSchema(name="content_type", dtype=DataType.VARCHAR,       max_length=50),
    FieldSchema(name="h1",           dtype=DataType.VARCHAR,       max_length=300),
    FieldSchema(name="h2",           dtype=DataType.VARCHAR,       max_length=300),
    FieldSchema(name="vector",       dtype=DataType.FLOAT_VECTOR,  dim=768),
]
schema     = CollectionSchema(fields, "Kubeflow manifests code index")
collection = Collection("code_index", schema)
print("Created code_index collection")

# Sample Kubernetes resources
sample_code = [
    {
        "text": """apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: inferenceservice.serving.kserve.io
webhooks:
  - name: inferenceservice.kserve-webhook-server.pod-mutator
    admissionReviewVersions: [v1beta1]
    clientConfig:
      service:
        name: kserve-webhook-server-service
        namespace: kserve
        path: /mutate-pods""",
        "url":  "https://github.com/kubeflow/manifests/blob/master/apps/kserve/webhook.yaml",
        "kind": "MutatingWebhookConfiguration",
        "name": "inferenceservice.serving.kserve.io",
        "type": "kubernetes_resource"
    },
    {
        "text": """apiVersion: apps/v1
kind: Deployment
metadata:
  name: kserve-controller-manager
  namespace: kserve
spec:
  replicas: 1
  selector:
    matchLabels:
      control-plane: kserve-controller-manager""",
        "url":  "https://github.com/kubeflow/manifests/blob/master/apps/kserve/deployment.yaml",
        "kind": "Deployment",
        "name": "kserve-controller-manager",
        "type": "kubernetes_resource"
    },
    {
        "text": """def create_inference_service(name: str, model_uri: str, namespace: str = 'default'):
    \"\"\"Creates a KServe InferenceService for model serving.\"\"\"
    from kubernetes import client
    isvc = {
        'apiVersion': 'serving.kserve.io/v1beta1',
        'kind': 'InferenceService',
        'metadata': {'name': name, 'namespace': namespace},
        'spec': {'predictor': {'model': {'modelFormat': {'name': 'sklearn'}, 'storageUri': model_uri}}}
    }
    return isvc""",
        "url":  "https://github.com/kubeflow/manifests/blob/master/apps/kserve/utils.py#L10",
        "kind": "FunctionDef",
        "name": "create_inference_service",
        "type": "python_definition"
    },
    {
        "text": """apiVersion: v1
kind: ConfigMap
metadata:
  name: inferenceservice-config
  namespace: kserve
data:
  agent: |
    {
      "image": "kserve/agent:latest",
      "memoryRequest": "100Mi",
      "memoryLimit": "1Gi"
    }""",
        "url":  "https://github.com/kubeflow/manifests/blob/master/apps/kserve/configmap.yaml",
        "kind": "ConfigMap",
        "name": "inferenceservice-config",
        "type": "kubernetes_resource"
    },
    {
        "text": """apiVersion: v1
kind: Service
metadata:
  name: kserve-webhook-server-service
  namespace: kserve
spec:
  ports:
    - port: 443
      targetPort: 9443
  selector:
    control-plane: kserve-controller-manager""",
        "url":  "https://github.com/kubeflow/manifests/blob/master/apps/kserve/service.yaml",
        "kind": "Service",
        "name": "kserve-webhook-server-service",
        "type": "kubernetes_resource"
    },
]

print(f"Embedding {len(sample_code)} code units...")
texts      = [d["text"] for d in sample_code]
embeddings = model.encode(texts).tolist()

collection.insert([{
    "content_text": d["text"],
    "source_url":   d["url"],
    "kind":         d["kind"],
    "name":         d["name"],
    "content_type": d["type"],
    "h1":           d["kind"],
    "h2":           d["name"],
    "vector":       e
} for d, e in zip(sample_code, embeddings)])

collection.flush()

index_params = {
    "metric_type": "COSINE",
    "index_type":  "IVF_FLAT",
    "params":      {"nlist": 128}
}
collection.create_index("vector", index_params)
collection.load()

print(f"Seeded code_index with {collection.num_entities} records")
print("Done - code_index ready")