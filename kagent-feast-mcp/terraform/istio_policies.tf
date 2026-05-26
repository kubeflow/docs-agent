# =============================================================================
# Istio AuthorizationPolicies
# 
# Default mesh policy is deny-all. These policies explicitly allow
# the traffic flows needed by our stack.
# =============================================================================

# --- Milvus internal traffic (ml-infra) ---

resource "kubectl_manifest" "istio_allow_milvus_standalone" {
  yaml_body = <<YAML
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-milvus-standalone
  namespace: ml-infra
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: milvus
      component: standalone
  action: ALLOW
  rules:
  - to:
    - operation:
        ports: ["19530", "9091"]
YAML

  depends_on = [helm_release.istiod, kubernetes_namespace.ml_infra]
}

resource "kubectl_manifest" "istio_allow_milvus_etcd" {
  yaml_body = <<YAML
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-milvus-etcd
  namespace: ml-infra
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: etcd
  action: ALLOW
  rules:
  - to:
    - operation:
        ports: ["2379", "2380"]
YAML

  depends_on = [helm_release.istiod, kubernetes_namespace.ml_infra]
}

resource "kubectl_manifest" "istio_allow_milvus_minio" {
  yaml_body = <<YAML
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-milvus-minio
  namespace: ml-infra
spec:
  selector:
    matchLabels:
      app: minio
  action: ALLOW
  rules:
  - to:
    - operation:
        ports: ["9000", "9001"]
YAML

  depends_on = [helm_release.istiod, kubernetes_namespace.ml_infra]
}

# --- Cross-namespace traffic ---

# Allow docs-agent/mcp-server -> ml-infra/milvus (vector DB queries)
resource "kubectl_manifest" "istio_allow_mcp_to_milvus" {
  yaml_body = <<YAML
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-mcp-to-milvus
  namespace: ml-infra
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: milvus
      component: standalone
  action: ALLOW
  rules:
  - from:
    - source:
        namespaces: ["docs-agent"]
    to:
    - operation:
        ports: ["19530"]
YAML

  depends_on = [helm_release.istiod, kubernetes_namespace.ml_infra, kubernetes_namespace.docs_agent]
}

# Allow docs-agent/kagent -> ml-infra/qwen-llm (LLM inference)
resource "kubectl_manifest" "istio_allow_kagent_to_llm" {
  yaml_body = <<YAML
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-kagent-to-llm
  namespace: ml-infra
spec:
  selector:
    matchLabels:
      serving.kserve.io/inferenceservice: qwen-llm
  action: ALLOW
  rules:
  - from:
    - source:
        namespaces: ["docs-agent"]
    to:
    - operation:
        ports: ["8080"]
YAML

  depends_on = [helm_release.istiod, kubernetes_namespace.ml_infra, kubernetes_namespace.docs_agent]
}

# Allow docs-agent/kagent -> docs-agent/mcp-server (MCP tool calls)
resource "kubectl_manifest" "istio_allow_kagent_to_mcp" {
  yaml_body = <<YAML
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-kagent-to-mcp
  namespace: docs-agent
spec:
  selector:
    matchLabels:
      app: mcp-kubeflow-docs
  action: ALLOW
  rules:
  - from:
    - source:
        namespaces: ["docs-agent"]
    to:
    - operation:
        ports: ["8000"]
YAML

  depends_on = [helm_release.istiod, kubernetes_namespace.docs_agent]
}
