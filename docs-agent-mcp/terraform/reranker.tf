# =============================================================================
# reranker.tf
# Deploys a cross-encoder model (e.g. BAAI/bge-reranker-base) as a CPU-based
# KServe InferenceService using Hugging Face Text Embeddings Inference (TEI).
# =============================================================================

resource "kubectl_manifest" "reranker_service" {
  yaml_body = <<YAML
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: reranker-service
  namespace: $${var.namespace_ml_infra}
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    containers:
      - name: kserve-container
        image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.7
        args:
          - --model-id
          - $${var.reranker_model_id}
          - --port
          - "8080"
          - --auto-truncate
          - --max-client-batch-size
          - "$${var.reranker_max_client_batch_size}"
          - --max-batch-tokens
          - "$${var.reranker_max_batch_tokens}"
        resources:
          requests:
            cpu: "100m"
            memory: "256Mi"
          limits:
            cpu: "2"
            memory: "4Gi"
        ports:
          - containerPort: 8080
            protocol: TCP
YAML

  depends_on = [kubernetes_namespace.ml_infra]
}
