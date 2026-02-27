# ---------------------------------------------------------------------------
# Input variables for the root module
# ---------------------------------------------------------------------------

# -- OCI authentication ----------------------------------------------------

variable "tenancy_ocid" {
  description = "OCID of the OCI tenancy."
  type        = string
}

variable "user_ocid" {
  description = "OCID of the OCI user."
  type        = string
}

variable "fingerprint" {
  description = "Fingerprint for the OCI API signing key."
  type        = string
}

variable "private_key" {
  description = "Private key content (or path via file()) for OCI API auth."
  type        = string
  sensitive   = true
}

variable "region" {
  description = "OCI region (e.g. us-ashburn-1)."
  type        = string
  default     = "us-ashburn-1"
}

variable "compartment_id" {
  description = "OCID of the compartment to deploy resources into."
  type        = string
}

# -- OKE cluster -----------------------------------------------------------

variable "cluster_name" {
  description = "Name for the OKE Kubernetes cluster."
  type        = string
  default     = "docs-agent-cluster"
}

variable "kubernetes_version" {
  description = "Kubernetes version for the OKE cluster."
  type        = string
  default     = "v1.28.2"
}

variable "vcn_cidr" {
  description = "CIDR block for the VCN."
  type        = string
  default     = "10.0.0.0/16"
}

# -- Node pools ------------------------------------------------------------

variable "cpu_node_shape" {
  description = "OCI shape for CPU worker nodes."
  type        = string
  default     = "VM.Standard.E4.Flex"
}

variable "cpu_node_count" {
  description = "Number of CPU worker nodes."
  type        = number
  default     = 3
}

variable "gpu_node_shape" {
  description = "OCI shape for GPU worker nodes (for KServe LLM inference)."
  type        = string
  default     = "VM.GPU.A10.1"
}

variable "gpu_node_count" {
  description = "Number of GPU worker nodes."
  type        = number
  default     = 1
}

# -- Application -----------------------------------------------------------

variable "namespace" {
  description = "Kubernetes namespace for the docs-agent stack."
  type        = string
  default     = "docs-agent"
}

variable "docs_agent_image" {
  description = "Container image for the docs-agent servers."
  type        = string
  default     = "ghcr.io/kubeflow/docs-agent:latest"
}

variable "docs_agent_replicas" {
  description = "Number of docs-agent pod replicas."
  type        = number
  default     = 2
}

variable "milvus_storage_gb" {
  description = "Persistent volume size in GB for Milvus data."
  type        = number
  default     = 50
}
