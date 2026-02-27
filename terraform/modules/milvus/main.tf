# ---------------------------------------------------------------------------
# Milvus Module — Deploy Milvus via Helm
# ---------------------------------------------------------------------------

terraform {
  required_providers {
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.11"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.23"
    }
  }
}

variable "namespace" {
  type    = string
  default = "docs-agent"
}

variable "milvus_storage" {
  description = "Storage size in GB for Milvus persistent volume."
  type        = number
  default     = 50
}

# -- Namespace --------------------------------------------------------------

resource "kubernetes_namespace" "this" {
  metadata {
    name = var.namespace
  }
}

# -- Milvus Helm Release ---------------------------------------------------

resource "helm_release" "milvus" {
  name       = "milvus"
  namespace  = kubernetes_namespace.this.metadata[0].name
  repository = "https://zilliztech.github.io/milvus-helm"
  chart      = "milvus"
  version    = "4.1.26"
  wait       = true
  timeout    = 600

  # Standalone mode (suitable for dev/small deployments)
  set {
    name  = "cluster.enabled"
    value = "false"
  }
  set {
    name  = "standalone.enabled"
    value = "true"
  }

  # Persistence
  set {
    name  = "standalone.persistence.enabled"
    value = "true"
  }
  set {
    name  = "standalone.persistence.size"
    value = "${var.milvus_storage}Gi"
  }

  # Expose on default port
  set {
    name  = "service.type"
    value = "ClusterIP"
  }
  set {
    name  = "service.port"
    value = "19530"
  }
}

# -- Outputs ---------------------------------------------------------------

# The Helm chart creates a service named "<release>-milvus" by default.
output "service_host" {
  value = "milvus-milvus.${var.namespace}.svc.cluster.local"
}

output "service_port" {
  value = "19530"
}
