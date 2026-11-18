# =============================================================================
# deployments/terraform/oke-cluster/variables.tf
# OCI OKE cluster + VCN + CPU & GPU node pools for kubeflow/docs-agent.
# Templated automatically by `docs-cli` from docs.toml + environment variables.
# =============================================================================

variable "compartment_ocid" {
  description = "OCID of the OCI compartment where the dev cluster and VCN will be created."
  type        = string
  validation {
    condition     = startswith(var.compartment_ocid, "ocid1.compartment.") || startswith(var.compartment_ocid, "ocid1.tenancy.")
    error_message = "compartment_ocid must start with ocid1.compartment. or ocid1.tenancy."
  }
}

variable "cluster_name" {
  description = "Name prefix for the OKE cluster, VCN, node pools, and freeform ownership tags."
  type        = string
  default     = "docs-agent-dev"
}

variable "region" {
  description = "OCI region (e.g. us-ashburn-1, us-chicago-1, us-phoenix-1)."
  type        = string
  default     = "us-ashburn-1"
}

variable "oci_profile" {
  description = "Profile in ~/.oci/config when not authenticating via explicit OCI env vars."
  type        = string
  default     = "DEFAULT"
}

# --- Optional explicit OCI API key auth (populated by docs-cli from OCI_* env vars) ---

variable "tenancy_ocid" {
  description = "OCI Tenancy OCID (optional when using ~/.oci/config)."
  type        = string
  default     = null
}

variable "user_ocid" {
  description = "OCI User OCID (optional when using ~/.oci/config)."
  type        = string
  default     = null
}

variable "fingerprint" {
  description = "OCI API key fingerprint (optional when using ~/.oci/config)."
  type        = string
  default     = null
}

variable "private_key_path" {
  description = "Path to OCI API private key PEM file (optional when using ~/.oci/config)."
  type        = string
  default     = null
}

variable "private_key_password" {
  description = "Passphrase for OCI API private key, if encrypted."
  type        = string
  default     = null
  sensitive   = true
}

# --- Kubernetes & Networking -------------------------------------------------

variable "kubernetes_version" {
  description = "Kubernetes version (e.g. \"v1.31.1\"). Empty = newest OKE version with matching CPU/GPU images."
  type        = string
  default     = ""
}

variable "api_allowed_cidr" {
  description = "CIDR allowed to reach the public Kubernetes API endpoint on 6443 (OCI IAM auth is always required)."
  type        = string
  default     = "0.0.0.0/0"
}

# --- CPU Node Pool (Milvus, KFP, TEI embeddings & reranker, Kagent, MCP) -----

variable "cpu_node_shape" {
  description = "OCI compute shape for CPU worker nodes (x86_64 Flex shape)."
  type        = string
  default     = "VM.Standard.E5.Flex"
}

variable "cpu_node_count" {
  description = "Number of worker nodes in the CPU node pool."
  type        = number
  default     = 2
}

variable "cpu_node_ocpus" {
  description = "OCPUs per CPU worker node (1 OCPU = 2 vCPUs)."
  type        = number
  default     = 4
}

variable "cpu_node_memory_gbs" {
  description = "Memory in GB per CPU worker node."
  type        = number
  default     = 32
}

variable "cpu_boot_volume_gbs" {
  description = "Boot volume size in GB per CPU worker node."
  type        = number
  default     = 100
}

# --- GPU Node Pool (vLLM via llm-runtime chart: Gemma 4 / Qwen) --------------

variable "enable_gpu_pool" {
  description = "Provision a GPU node pool for in-cluster vLLM inference (set false for --no-gpu / external LLM mode)."
  type        = bool
  default     = true
}

variable "gpu_node_shape" {
  description = "OCI GPU shape for LLM inference (e.g. VM.GPU.A10.1)."
  type        = string
  default     = "VM.GPU.A10.1"
}

variable "gpu_node_count" {
  description = "Number of GPU worker nodes."
  type        = number
  default     = 1
}

variable "gpu_node_ocpus" {
  description = "OCPUs for Flex GPU shapes (ignored for fixed shapes like VM.GPU.A10.1)."
  type        = number
  default     = 15
}

variable "gpu_node_memory_gbs" {
  description = "Memory in GB for Flex GPU shapes (ignored for fixed shapes like VM.GPU.A10.1)."
  type        = number
  default     = 240
}

variable "gpu_boot_volume_gbs" {
  description = "Boot volume size in GB per GPU worker node (250 GB recommended for container images + HF cache)."
  type        = number
  default     = 250
}

variable "gpu_availability_domains" {
  description = "Optional list of Availability Domains with verified GPU host capacity (auto-populated via OCI Capacity Report)."
  type        = list(string)
  default     = []
}

variable "ssh_public_key" {
  description = "Optional SSH public key for worker nodes."
  type        = string
  default     = ""
}
