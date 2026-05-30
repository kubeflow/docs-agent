# =============================================================================
# kagent.tf
# Installs kagent CRDs and the kagent controller/UI into the docs-agent
# namespace using the official OCI Helm charts from GHCR.
#
# Two-chart pattern (kagent-crds then kagent) mirrors the upstream install
# guide and ensures CRDs are established before the controller starts.
# No version pin — latest is used so we always pull the newest stable release.
# =============================================================================

# Step 1: Install kagent CRDs (ModelConfig, RemoteMCPServer, Agent, etc.)
resource "helm_release" "kagent_crds" {
  name             = "kagent-crds"
  repository       = "oci://ghcr.io/kagent-dev/kagent/helm"
  chart            = "kagent-crds"
  namespace        = var.namespace_docs_agent
  create_namespace = false # namespace already managed in namespaces.tf

  depends_on = [kubernetes_namespace.docs_agent]
}

# Step 2: Install the kagent controller and UI.
# Built-in bundled agents and extra tools are disabled — we only need the
# core controller to process our custom Agent CR defined in setup.yaml.
resource "helm_release" "kagent" {
  name             = "kagent"
  repository       = "oci://ghcr.io/kagent-dev/kagent/helm"
  chart            = "kagent"
  namespace        = var.namespace_docs_agent
  create_namespace = false

  # Disable all pre-bundled agents — we bring our own via setup.yaml
  set {
    name  = "argo-rollouts-agent.enabled"
    value = "false"
  }
  set {
    name  = "cilium-debug-agent.enabled"
    value = "false"
  }
  set {
    name  = "cilium-manager-agent.enabled"
    value = "false"
  }
  set {
    name  = "cilium-policy-agent.enabled"
    value = "false"
  }
  set {
    name  = "helm-agent.enabled"
    value = "false"
  }
  set {
    name  = "istio-agent.enabled"
    value = "false"
  }
  set {
    name  = "k8s-agent.enabled"
    value = "false"
  }
  set {
    name  = "kgateway-agent.enabled"
    value = "false"
  }
  set {
    name  = "observability-agent.enabled"
    value = "false"
  }
  set {
    name  = "promql-agent.enabled"
    value = "false"
  }
  set {
    name  = "grafana-mcp.enabled"
    value = "false"
  }
  set {
    name  = "querydoc.enabled"
    value = "false"
  }

  # Lower resource requests to absolute minimum for dev cluster CPU nodes
  set {
    name  = "database.postgres.bundled.resources.requests.cpu"
    value = "100m"
  }
  set {
    name  = "database.postgres.bundled.resources.requests.memory"
    value = "128Mi"
  }
  set {
    name  = "controller.resources.requests.cpu"
    value = "100m"
  }
  set {
    name  = "controller.resources.requests.memory"
    value = "128Mi"
  }
  set {
    name  = "ui.resources.requests.cpu"
    value = "50m"
  }
  set {
    name  = "ui.resources.requests.memory"
    value = "128Mi"
  }
  set {
    name  = "kagent-tools.tools.resources.requests.cpu"
    value = "50m"
  }
  set {
    name  = "kagent-tools.tools.resources.requests.memory"
    value = "64Mi"
  }

  depends_on = [helm_release.kagent_crds]
}
