# =============================================================================
# namespaces.tf
# Creates all application namespaces managed by Terraform.
# Workload CRDs (InferenceService, Agent, etc.) are deployed via CD — only
# the namespace lifecycle is owned here.
# =============================================================================

resource "kubernetes_namespace" "ml_infra" {
  metadata {
    name = var.namespace_ml_infra
  }
}

resource "kubernetes_namespace" "docs_agent" {
  metadata {
    name = var.namespace_docs_agent
  }
}
