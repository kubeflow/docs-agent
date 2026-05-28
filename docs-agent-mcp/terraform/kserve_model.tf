# Namespace for ML workloads — created by Terraform,
# workload CRDs (ServingRuntime, InferenceService) deployed via GitHub Actions CD.
resource "kubernetes_namespace" "ml_infra" {
  metadata {
    name = "ml-infra"
  }
}
resource "kubernetes_namespace" "docs_agent" {
  metadata {
    name = "docs-agent"
  }
}
