# =============================================================================
# kagent_ingress.tf
# Optional public Istio Gateway + Let's Encrypt TLS for Kagent UI / A2A.
# Disabled by default — enable when DNS points at the cluster ingress IP.
# Clusters using only kagent-ui-lb (OCI LoadBalancer) can leave this off.
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

variable "kagent_cors_allow_origin_regexes" {
  description = "Istio CORS allowOrigins regex list (e.g. Vercel preview apps, kubeflow.org)"
  type        = list(string)
  default     = [".*"]
}

locals {
  kagent_cors_origins_yaml = join("\n", [
    for r in var.kagent_cors_allow_origin_regexes : "            - regex: \"${r}\""
  ])
}

resource "kubectl_manifest" "cluster_issuer_letsencrypt" {
  count = var.enable_kagent_ingress ? 1 : 0

  yaml_body = <<-YAML
    apiVersion: cert-manager.io/v1
    kind: ClusterIssuer
    metadata:
      name: letsencrypt-prod
    spec:
      acme:
        server: https://acme-v02.api.letsencrypt.org/directory
        email: ${var.kagent_acme_email}
        privateKeySecretRef:
          name: letsencrypt-prod-account-key
        solvers:
          - http01:
              ingress:
                ingressClassName: istio
  YAML

  depends_on = [helm_release.cert_manager]
}

resource "kubectl_manifest" "kagent_ui_certificate" {
  count = var.enable_kagent_ingress ? 1 : 0

  yaml_body = <<-YAML
    apiVersion: cert-manager.io/v1
    kind: Certificate
    metadata:
      name: kagent-ui-cert
      namespace: istio-system
    spec:
      secretName: kagent-ui-tls
      issuerRef:
        name: letsencrypt-prod
        kind: ClusterIssuer
      commonName: ${var.kagent_domain_name}
      dnsNames:
        - ${var.kagent_domain_name}
  YAML

  depends_on = [kubectl_manifest.cluster_issuer_letsencrypt]
}

resource "kubectl_manifest" "kagent_gateway" {
  count = var.enable_kagent_ingress ? 1 : 0

  yaml_body = <<-YAML
    apiVersion: networking.istio.io/v1alpha3
    kind: Gateway
    metadata:
      name: kagent-gateway
      namespace: istio-system
    spec:
      selector:
        istio: ingressgateway
      servers:
        - port:
            number: 80
            name: http
            protocol: HTTP
          hosts:
            - "${var.kagent_domain_name}"
          tls:
            httpsRedirect: true
        - port:
            number: 443
            name: https
            protocol: HTTPS
          hosts:
            - "${var.kagent_domain_name}"
          tls:
            mode: SIMPLE
            credentialName: kagent-ui-tls
  YAML

  depends_on = [helm_release.istiod]
}

resource "kubectl_manifest" "kagent_virtualservice" {
  count = var.enable_kagent_ingress ? 1 : 0

  yaml_body = <<-YAML
    apiVersion: networking.istio.io/v1alpha3
    kind: VirtualService
    metadata:
      name: kagent-ui-routing
      namespace: ${var.namespace_docs_agent}
    spec:
      hosts:
        - "${var.kagent_domain_name}"
      gateways:
        - istio-system/kagent-gateway
      http:
        - corsPolicy:
            allowOrigins:
${local.kagent_cors_origins_yaml}
            allowMethods:
              - POST
              - GET
              - OPTIONS
            allowHeaders:
              - "*"
          route:
            - destination:
                host: kagent-ui.${var.namespace_docs_agent}.svc.cluster.local
                port:
                  number: 8080
  YAML

  depends_on = [kubectl_manifest.kagent_gateway, helm_release.kagent]
}
