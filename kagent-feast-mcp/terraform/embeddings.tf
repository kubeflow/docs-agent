# =============================================================================
# embeddings.tf
# Deploys the sentence-transformers/all-mpnet-base-v2 model as a CPU-based
# KServe InferenceService using Hugging Face Text Embeddings Inference (TEI).
# =============================================================================

resource "kubectl_manifest" "embeddings_service" {
  yaml_body = <<YAML
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: embeddings-service
  namespace: ${var.namespace_ml_infra}
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    containers:
      - name: kserve-container
        image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.7
        args:
          - --model-id
          - ${var.embeddings_model_id}
          - --port
          - "8080"
          # Truncate inputs to model max tokens instead of HTTP 413 (mpnet = 384 tokens).
          - --auto-truncate
          - --max-client-batch-size
          - "${var.embeddings_max_client_batch_size}"
          - --max-batch-tokens
          - "${var.embeddings_max_batch_tokens}"
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
