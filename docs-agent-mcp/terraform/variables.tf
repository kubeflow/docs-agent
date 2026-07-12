# =============================================================================
# variables.tf
# Central source of truth for all versions, namespace names, and resource
# limits used across the stack. Change a value here and it propagates
# to every file that references it — no need to hunt through multiple TF files.
# =============================================================================

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
  description = "Kubeflow Pipelines standalone version"
  type        = string
  default     = "2.15.0"
}

variable "milvus_version" {
  description = "Milvus standalone container image version (used in Milvus CR)"
  type        = string
  default     = "v2.4.15"
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
