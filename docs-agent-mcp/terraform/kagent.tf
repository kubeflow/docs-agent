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
  set { name = "agents.argo-rollouts-agent.enabled";  value = "false" }
  set { name = "agents.cilium-debug-agent.enabled";   value = "false" }
  set { name = "agents.cilium-manager-agent.enabled"; value = "false" }
  set { name = "agents.cilium-policy-agent.enabled";  value = "false" }
  set { name = "agents.helm-agent.enabled";           value = "false" }
  set { name = "agents.istio-agent.enabled";          value = "false" }
  set { name = "agents.k8s-agent.enabled";            value = "false" }
  set { name = "agents.kgateway-agent.enabled";       value = "false" }
  set { name = "agents.observability-agent.enabled";  value = "false" }
  set { name = "agents.promql-agent.enabled";         value = "false" }
  set { name = "tools.grafana-mcp.enabled";           value = "false" }
  set { name = "tools.querydoc.enabled";              value = "false" }

  depends_on = [helm_release.kagent_crds]
}
