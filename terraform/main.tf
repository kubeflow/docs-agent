# ---------------------------------------------------------------------------
# Kubeflow Docs-Agent — OCI Terraform Root Module
# ---------------------------------------------------------------------------

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.11"
    }
  }

  # Uncomment to use OCI Object Storage as remote backend:
  # backend "s3" {
  #   bucket   = "docs-agent-tfstate"
  #   key      = "terraform.tfstate"
  #   region   = "us-ashburn-1"
  #   endpoint = "https://<namespace>.compat.objectstorage.<region>.oraclecloud.com"
  # }
}

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

provider "oci" {
  tenancy_ocid = var.tenancy_ocid
  user_ocid    = var.user_ocid
  fingerprint  = var.fingerprint
  private_key  = var.private_key
  region       = var.region
}

provider "kubernetes" {
  host                   = module.oke.cluster_endpoint
  cluster_ca_certificate = module.oke.cluster_ca_certificate
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "oci"
    args = [
      "ce", "cluster", "generate-token",
      "--cluster-id", module.oke.cluster_id,
      "--region", var.region,
    ]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.oke.cluster_endpoint
    cluster_ca_certificate = module.oke.cluster_ca_certificate
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "oci"
      args = [
        "ce", "cluster", "generate-token",
        "--cluster-id", module.oke.cluster_id,
        "--region", var.region,
      ]
    }
  }
}

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

module "oke" {
  source = "./modules/oke"

  compartment_id    = var.compartment_id
  cluster_name      = var.cluster_name
  kubernetes_version = var.kubernetes_version
  vcn_cidr          = var.vcn_cidr
  region            = var.region

  # Node pools
  cpu_node_shape    = var.cpu_node_shape
  cpu_node_count    = var.cpu_node_count
  gpu_node_shape    = var.gpu_node_shape
  gpu_node_count    = var.gpu_node_count
}

module "milvus" {
  source     = "./modules/milvus"
  depends_on = [module.oke]

  namespace        = var.namespace
  milvus_storage   = var.milvus_storage_gb
}

module "kserve" {
  source     = "./modules/kserve"
  depends_on = [module.oke]

  namespace = var.namespace
}

module "docs_agent" {
  source     = "./modules/docs-agent"
  depends_on = [module.milvus, module.kserve]

  namespace    = var.namespace
  image        = var.docs_agent_image
  replicas     = var.docs_agent_replicas
  milvus_host  = module.milvus.service_host
  milvus_port  = module.milvus.service_port
  kserve_url   = module.kserve.inference_url
}
