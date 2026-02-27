# ---------------------------------------------------------------------------
# Docs-Agent Module — Deploy the WebSocket + HTTPS API servers
# ---------------------------------------------------------------------------
#
# The container image should be built from the repository root so that
# both server/ and shared/ directories are in the build context:
#   docker build -f server/Dockerfile -t docs-agent:latest .

terraform {
  required_providers {
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

variable "image" {
  type    = string
  default = "ghcr.io/kubeflow/docs-agent:latest"
}

variable "replicas" {
  type    = number
  default = 2
}

variable "milvus_host" {
  type = string
}

variable "milvus_port" {
  type    = string
  default = "19530"
}

variable "kserve_url" {
  type = string
}

# -- WebSocket server (port 8765) ------------------------------------------

resource "kubernetes_deployment" "ws_server" {
  metadata {
    name      = "docs-agent-ws"
    namespace = var.namespace
    labels    = { app = "docs-agent-ws" }
  }

  spec {
    replicas = var.replicas

    selector {
      match_labels = { app = "docs-agent-ws" }
    }

    template {
      metadata {
        labels = { app = "docs-agent-ws" }
      }

      spec {
        container {
          name  = "ws-server"
          image = var.image

          command = ["python", "server/app.py"]

          port {
            container_port = 8765
          }

          env {
            name  = "MILVUS_HOST"
            value = var.milvus_host
          }
          env {
            name  = "MILVUS_PORT"
            value = var.milvus_port
          }
          env {
            name  = "KSERVE_URL"
            value = var.kserve_url
          }
          env {
            name  = "PORT"
            value = "8765"
          }

          resources {
            requests = {
              cpu    = "250m"
              memory = "512Mi"
            }
            limits = {
              cpu    = "1"
              memory = "2Gi"
            }
          }

          liveness_probe {
            tcp_socket {
              port = 8765
            }
            initial_delay_seconds = 10
            period_seconds        = 30
          }

          readiness_probe {
            tcp_socket {
              port = 8765
            }
            initial_delay_seconds = 5
            period_seconds        = 10
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "ws_server" {
  metadata {
    name      = "docs-agent-ws"
    namespace = var.namespace
  }

  spec {
    selector = { app = "docs-agent-ws" }

    port {
      port        = 8765
      target_port = 8765
    }

    type = "ClusterIP"
  }
}

# -- HTTPS server (port 8000) ---------------------------------------------

resource "kubernetes_deployment" "https_server" {
  metadata {
    name      = "docs-agent-https"
    namespace = var.namespace
    labels    = { app = "docs-agent-https" }
  }

  spec {
    replicas = var.replicas

    selector {
      match_labels = { app = "docs-agent-https" }
    }

    template {
      metadata {
        labels = { app = "docs-agent-https" }
      }

      spec {
        container {
          name  = "https-server"
          image = var.image

          command = ["python", "server-https/app.py"]

          port {
            container_port = 8000
          }

          env {
            name  = "MILVUS_HOST"
            value = var.milvus_host
          }
          env {
            name  = "MILVUS_PORT"
            value = var.milvus_port
          }
          env {
            name  = "KSERVE_URL"
            value = var.kserve_url
          }
          env {
            name  = "PORT"
            value = "8000"
          }

          resources {
            requests = {
              cpu    = "250m"
              memory = "512Mi"
            }
            limits = {
              cpu    = "1"
              memory = "2Gi"
            }
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 10
            period_seconds        = 30
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 5
            period_seconds        = 10
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "https_server" {
  metadata {
    name      = "docs-agent-https"
    namespace = var.namespace
  }

  spec {
    selector = { app = "docs-agent-https" }

    port {
      port        = 8000
      target_port = 8000
    }

    type = "ClusterIP"
  }
}

# -- Outputs ---------------------------------------------------------------

output "service_url" {
  value = "http://docs-agent-https.${var.namespace}.svc.cluster.local:8000"
}
