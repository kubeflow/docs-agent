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

variable "enable_session_auth" {
  description = "Deploy the anonymous session-JWT issuer + Istio RequestAuthentication"
  type        = bool
  default     = true
}

variable "enforce_session_auth" {
  description = "Hard-require a session JWT on the A2A paths (flip only after the widget attaches tokens)"
  type        = bool
  default     = false
}

variable "session_issuer_image" {
  description = "Container image for the session-token issuer"
  type        = string
  default     = "ghcr.io/kubeflow/kubeflow-session-issuer:latest"
}

# RS256 signing key for session JWTs, generated in-state and delivered as the
# Secret the issuer Deployment mounts. Rotation = taint this resource, which
# also invalidates every outstanding session token.
resource "tls_private_key" "session_issuer" {
  count     = var.enable_session_auth ? 1 : 0
  algorithm = "RSA"
  rsa_bits  = 2048
}

resource "kubernetes_secret" "session_issuer_key" {
  count = var.enable_session_auth ? 1 : 0

  metadata {
    name      = "session-issuer-key"
    namespace = var.namespace_docs_agent
  }

  data = {
    "private.pem" = tls_private_key.session_issuer[0].private_key_pem
  }

  depends_on = [kubernetes_namespace.docs_agent]
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
      sessionAuth = {
        enabled = var.enable_session_auth
        enforce = var.enforce_session_auth
        image   = var.session_issuer_image
      }
    })
  ]

  depends_on = [
    helm_release.istiod,
    helm_release.cert_manager,
    helm_release.kagent,
    kubernetes_namespace.docs_agent,
    kubernetes_namespace.ml_infra,
    kubernetes_secret.session_issuer_key,
  ]
}
