# =============================================================================
# deployments/terraform/gke-cluster/variables.tf
# Google Cloud GKE cluster + VPC + CPU & GPU node pools for kubeflow/docs-agent.
# Templated automatically by `docs-cli --cloud gcp` from docs.toml + env vars.
# =============================================================================

variable "project_id" {
  description = "Google Cloud Project ID where the GKE cluster and VPC will be provisioned."
  type        = string
  validation {
    condition     = length(trimspace(var.project_id)) > 0
    error_message = "project_id must be a non-empty GCP Project ID."
  }
}

variable "cluster_name" {
  description = "Name prefix for the GKE cluster, VPC, node pools, and ownership labels."
  type        = string
  default     = "docs-agent-dev"
}

variable "region" {
  description = "GCP region for the VPC and subnet (e.g. us-central1, us-east4, us-west1)."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Primary GCP zone for the GKE cluster and worker pools (e.g. us-central1-a)."
  type        = string
  default     = "us-central1-a"
}

variable "credentials_file" {
  description = "Optional path to a GCP Service Account JSON key file (omit when using ADC or Workload Identity)."
  type        = string
  default     = null
}

# --- Kubernetes & Networking -------------------------------------------------

variable "kubernetes_version" {
  description = "Optional Kubernetes version prefix for GKE (empty uses REGULAR release channel default)."
  type        = string
  default     = ""
}

variable "vpc_cidr" {
  description = "Primary IPv4 CIDR for the GKE node subnet."
  type        = string
  default     = "10.0.0.0/16"
}

variable "pods_cidr" {
  description = "Secondary IPv4 CIDR range for GKE VPC-native Pod IPs."
  type        = string
  default     = "10.48.0.0/14"
}

variable "services_cidr" {
  description = "Secondary IPv4 CIDR range for GKE VPC-native Service ClusterIPs."
  type        = string
  default     = "10.52.0.0/20"
}

variable "api_allowed_cidr" {
  description = "CIDR allowed to reach the public Kubernetes API endpoint (GCP IAM auth is always required)."
  type        = string
  default     = "0.0.0.0/0"
}

# --- CPU Node Pool (Milvus, KFP, TEI embeddings & reranker, Kagent, MCP) -----

variable "cpu_machine_type" {
  description = "GCE machine type for CPU worker nodes (2x n2-highmem-4 = 8 vCPU, 64 GB RAM total; fits new-project CPUS_ALL_REGIONS=12 quota)."
  type        = string
  default     = "n2-highmem-4"
}

variable "cpu_node_count" {
  description = "Number of worker nodes in the CPU node pool."
  type        = number
  default     = 2
}

variable "cpu_disk_size_gb" {
  description = "Boot persistent disk size in GB per CPU worker node."
  type        = number
  default     = 100
}

variable "cpu_disk_type" {
  description = "Boot persistent disk type for CPU worker nodes (pd-standard preserves new-project 250GB SSD_TOTAL_GB quota for PVCs)."
  type        = string
  default     = "pd-standard"
}

# --- GPU Node Pool (vLLM via llm-runtime chart: Gemma 4 / Qwen) --------------

variable "enable_gpu_pool" {
  description = "Provision a GPU node pool for in-cluster vLLM inference (set false for --no-gpu / external LLM mode)."
  type        = bool
  default     = true
}

variable "gpu_machine_type" {
  description = "GCE GPU machine type for LLM inference (g2-standard-12 = 1x NVIDIA L4 24GB VRAM, 12 vCPU, 48 GB RAM)."
  type        = string
  default     = "g2-standard-12"
}

variable "gpu_accelerator_type" {
  description = "GKE guest accelerator type (e.g. nvidia-l4, nvidia-tesla-a100)."
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_accelerator_count" {
  description = "Number of GPUs per GPU worker node."
  type        = number
  default     = 1
}

variable "gpu_node_count" {
  description = "Number of GPU worker nodes."
  type        = number
  default     = 1
}

variable "gpu_disk_size_gb" {
  description = "Boot persistent disk size in GB per GPU worker node (200 GB+ recommended for images + HF cache)."
  type        = number
  default     = 200
}

variable "gpu_disk_type" {
  description = "Boot persistent disk type for GPU worker nodes."
  type        = string
  default     = "pd-balanced"
}

variable "gpu_zones" {
  description = "Optional list of GCP zones in `region` with verified GPU availability (defaults to [var.zone])."
  type        = list(string)
  default     = []
}
