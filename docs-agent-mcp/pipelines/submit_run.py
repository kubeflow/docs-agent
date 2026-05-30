"""
Submit and run the github_rag_pipeline on the local KFP API server.
Port-forward must be active: kubectl port-forward svc/ml-pipeline 8888:8888 -n kubeflow
"""
import kfp

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
        "chunk_size":       500,
        "chunk_overlap":    50,
        "embeddings_service_url": "http://embeddings-service-predictor.ml-infra.svc.cluster.local/embed",
        "milvus_uri":       "http://milvus-milvus.ml-infra.svc.cluster.local:19530",
        "collection_name":  "kubeflow_docs",
    },
    run_name="kubeflow-docs-rag-run-1",
    experiment_name="kubeflow-docs-rag",
    enable_caching=False,
)

print(f"Run submitted!")
print(f"Run ID  : {run.run_id}")
print(f"View at : {KFP_HOST}/#/runs/details/{run.run_id}")
