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
          - sentence-transformers/all-mpnet-base-v2
          - --port
          - "8080"
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

  # Depend on ml-infra namespace being ready
  depends_on = [kubernetes_namespace.ml_infra]
}
