"""
Submit and run the github_rag_pipeline on the local KFP API server.
Port-forward must be active: kubectl port-forward svc/ml-pipeline 8888:8888 -n kubeflow
"""
import kfp

from utils import (
    DEFAULT_DOCS_CHUNK_OVERLAP,
    DEFAULT_DOCS_CHUNK_SIZE,
    DEFAULT_DOCS_MAX_TEI_CHARS,
    DEFAULT_EMBEDDING_DIM,
    DOCS_COLLECTION,
)

KFP_HOST = "http://localhost:8888"
PIPELINE_YAML = "github_rag_pipeline.yaml"

client = kfp.Client(host=KFP_HOST)

run = client.create_run_from_pipeline_package(
    pipeline_file=PIPELINE_YAML,
    arguments={
        "repo_owner":       "kubeflow",
        "repo_name":        "website",
        "directory_path":   "content/en/docs",
        "github_token":     "",          # pass a token if you hit rate limits
        "base_url":         "https://www.kubeflow.org/docs",
        "chunk_size":       DEFAULT_DOCS_CHUNK_SIZE,
        "chunk_overlap":    DEFAULT_DOCS_CHUNK_OVERLAP,
        "max_tei_chars":    DEFAULT_DOCS_MAX_TEI_CHARS,
        "embedding_dim":    DEFAULT_EMBEDDING_DIM,
        "milvus_host":      "milvus-milvus.ml-infra.svc.cluster.local",
        "milvus_port":      "19530",
        "collection_name":  DOCS_COLLECTION,
    },
    run_name="kubeflow-docs-rag-run-1",
    experiment_name="kubeflow-docs-rag",
    enable_caching=False,
)

print("Run submitted!")
print(f"Run ID  : {run.run_id}")
print(f"View at : {KFP_HOST}/#/runs/details/{run.run_id}")
