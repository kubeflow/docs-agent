# =============================================================================
# gateway_guardrails.tf
# Public gateway + guardrails for the chatbot, installed from the local Helm
# chart (../charts/gateway-guardrails). Replaces the raw kubectl_manifest
# heredocs that previously lived in istio_policies.tf and kagent_ingress.tf —
# the chart is now the single source of truth for the Istio edge config.
#
# MIGRATION NOTE (existing state): the old kubectl_manifest resources were
# removed from this module. On an already-provisioned cluster, either
# `terraform state rm` the old resources and let Helm adopt the live objects,
# or accept a destroy/recreate of the edge config (brief public blip).
# =============================================================================

variable "enable_kagent_ingress" {
  description = "Create Istio Gateway, Certificate, and VirtualService for public Kagent UI"
  type        = bool
  default     = false
}

variable "kagent_domain_name" {
  description = "FQDN for Kagent UI and A2A (must resolve to Istio ingress when enabled)"
  type        = string
  default     = "agent.example.com"
}

variable "kagent_acme_email" {
  description = "Email for Let's Encrypt expiry notifications"
  type        = string
  default     = "admin@example.com"
}

variable "kagent_cors_allow_origins" {
  description = "Exact origins allowed to embed the chatbot (no wildcards — CORS lockdown)"
  type        = list(string)
  default     = ["https://kubeflowdemochatbot.netlify.app"]
}

variable "a2a_global_rate_limit_rpm" {
  description = "Global requests/minute cap on the chatbot HTTPS listener (per ingress pod)"
  type        = number
  default     = 60
}

resource "helm_release" "gateway_guardrails" {
  name      = "gateway-guardrails"
  chart     = "${path.module}/../charts/gateway-guardrails"
  namespace = var.namespace_docs_agent

  values = [
    yamlencode({
      domain = var.kagent_domain_name
      namespaces = {
        docsAgent   = var.namespace_docs_agent
        mlInfra     = var.namespace_ml_infra
        kubeflow    = var.namespace_kubeflow
        istioSystem = "istio-system"
      }
      gateway = {
        enabled = var.enable_kagent_ingress
        tls = {
          acme = {
            email = var.kagent_acme_email
          }
        }
      }
      routing = {
        cors = {
          allowOrigins = [for o in var.kagent_cors_allow_origins : { exact = o }]
        }
      }
      rateLimit = {
        global = {
          requestsPerMinute = var.a2a_global_rate_limit_rpm
        }
      }
    })
  ]

  depends_on = [
    helm_release.istiod,
    helm_release.cert_manager,
    helm_release.kagent,
    kubernetes_namespace.docs_agent,
    kubernetes_namespace.ml_infra,
  ]
}
