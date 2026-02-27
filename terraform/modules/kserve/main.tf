# ---------------------------------------------------------------------------
# KServe Module — Install KServe + dependencies for LLM serving
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

variable "inference_service_name" {
  description = "Name of the KServe InferenceService resource for the LLM."
  type        = string
  default     = "llama"
}

# -- cert-manager (required by KServe) ------------------------------------

resource "helm_release" "cert_manager" {
  name             = "cert-manager"
  namespace        = "cert-manager"
  create_namespace = true
  repository       = "https://charts.jetstack.io"
  chart            = "cert-manager"
  version          = "v1.16.3"
  wait             = true
  timeout          = 300

  set {
    name  = "installCRDs"
    value = "true"
  }
}

# -- Knative Serving (required by KServe for scale-to-zero) ----------------

resource "helm_release" "knative_serving" {
  name             = "knative-serving"
  namespace        = "knative-serving"
  create_namespace = true
  repository       = "https://knative.github.io/operator"
  chart            = "knative-serving"
  version          = "1.12.0"
  wait             = true
  timeout          = 600

  depends_on = [helm_release.cert_manager]
}

# -- KServe ----------------------------------------------------------------

resource "helm_release" "kserve" {
  name             = "kserve"
  namespace        = "kserve"
  create_namespace = true
  repository       = "https://kserve.github.io/kserve"
  chart            = "kserve"
  version          = "0.12.0"
  wait             = true
  timeout          = 600

  set {
    name  = "kserve.controller.deploymentMode"
    value = "Serverless"
  }

  depends_on = [helm_release.knative_serving]
}

# -- Outputs ---------------------------------------------------------------

output "inference_url" {
  description = "KServe inference endpoint for the LLM InferenceService."
  value       = "http://${var.inference_service_name}.${var.namespace}.svc.cluster.local/openai/v1/chat/completions"
}
