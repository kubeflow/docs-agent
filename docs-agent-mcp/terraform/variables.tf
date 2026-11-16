# =============================================================================
# variables.tf
# Central source of truth for all versions, namespace names, and resource
# limits used across the stack. Change a value here and it propagates
# to every file that references it — no need to hunt through multiple TF files.
# =============================================================================

# --- Kubeconfig & Ingress Gateway --------------------------------------------

variable "kubeconfig_path" {
  description = "Path to kubeconfig file used by kubernetes, helm, and kubectl providers"
  type        = string
  default     = "~/.kube/config"
}

variable "enable_istio_ingressgateway" {
  description = "Install the Istio ingressgateway Helm release in istio-system (required by gateway-guardrails)"
  type        = bool
  default     = true
}

variable "oci_lb_min_bandwidth_mbps" {
  description = "Minimum bandwidth in Mbps for the OCI flexible LoadBalancer backing istio-ingressgateway"
  type        = string
  default     = "10"
}

variable "oci_lb_max_bandwidth_mbps" {
  description = "Maximum bandwidth in Mbps for the OCI flexible LoadBalancer backing istio-ingressgateway"
  type        = string
  default     = "100"
}

# --- Namespace Names ---------------------------------------------------------

variable "namespace_ml_infra" {
  description = "Namespace for ML inference workloads (KServe InferenceServices, Milvus)"
  type        = string
  default     = "ml-infra"
}

variable "namespace_docs_agent" {
  description = "Namespace for the docs-agent application (kagent, MCP server)"
  type        = string
  default     = "docs-agent"
}

variable "namespace_kubeflow" {
  description = "Namespace for Kubeflow Pipelines standalone"
  type        = string
  default     = "kubeflow"
}

# --- Component Versions ------------------------------------------------------

variable "cert_manager_version" {
  description = "cert-manager Helm chart version"
  type        = string
  default     = "v1.20.2"
}

variable "istio_version" {
  description = "Istio Helm chart version (base + istiod)"
  type        = string
  default     = "1.23.0"
}

variable "knative_version" {
  description = "Knative Serving release version (serving-crds, serving-core, net-istio)"
  type        = string
  default     = "1.22.0"
}

variable "kserve_version" {
  description = "KServe Helm chart version"
  type        = string
  default     = "v0.17.0"
}

variable "kfp_version" {
  description = "Kubeflow Pipelines standalone version (used when use_kubeflow_helm_charts is false)"
  type        = string
  default     = "2.15.0"
}

variable "use_kubeflow_helm_charts" {
  description = "Deploy Kubeflow components using the new experimental Kubeflow Helm charts from kubeflow/manifests instead of raw Kustomize"
  type        = bool
  default     = true
}

variable "kubeflow_pipelines_scenario" {
  description = "Scenario for the experimental Kubeflow Pipelines Helm chart (platform-database or platform-k8s-native)"
  type        = string
  default     = "platform-database"
}

variable "enable_kubeflow_full_platform_helm" {
  description = "Also deploy additional experimental Kubeflow Helm charts (Dashboard, Notebooks, Katib, Trainer, Hub, KServe UI)"
  type        = bool
  default     = false
}

variable "milvus_version" {
  description = "Milvus standalone container image version (used in Milvus CR)"
  type        = string
  default     = "v2.6.22"
}

# --- Ingress & Routing ---------------------------------------------------------
# (moved to gateway_guardrails.tf — the guardrails Helm chart owns the edge config)

# --- TEI Embeddings (ml-infra) -----------------------------------------------

variable "embeddings_model_id" {
  description = "Hugging Face model ID served by TEI (must match pipeline/MCP embedding dimension)"
  type        = string
  default     = "sentence-transformers/all-mpnet-base-v2"
}

variable "embeddings_max_client_batch_size" {
  description = "TEI --max-client-batch-size (number of client requests batched per forward pass)"
  type        = string
  default     = "64"
}

variable "embeddings_max_batch_tokens" {
  description = "TEI --max-batch-tokens (token budget per batched forward pass)"
  type        = string
  default     = "16384"
}

# --- TEI Reranker (ml-infra) -------------------------------------------------

variable "reranker_model_id" {
  description = "Hugging Face cross-encoder model ID served by TEI"
  type        = string
  default     = "BAAI/bge-reranker-base"
}

variable "reranker_max_client_batch_size" {
  description = "TEI --max-client-batch-size for reranker"
  type        = string
  default     = "32"
}

variable "reranker_max_batch_tokens" {
  description = "TEI --max-batch-tokens for reranker"
  type        = string
  default     = "16384"
}
