# =============================================================================
# kagent_ingress.tf
# Provisions the Istio Gateway, VirtualService, and Let's Encrypt Certificate
# required to securely expose the KAgent UI on the public internet.
# =============================================================================

resource "kubectl_manifest" "cluster_issuer_letsencrypt" {
  yaml_body = <<-YAML
    apiVersion: cert-manager.io/v1
    kind: ClusterIssuer
    metadata:
      name: letsencrypt-prod
    spec:
      acme:
        server: https://acme-v02.api.letsencrypt.org/directory
        email: ${var.acme_email}
        privateKeySecretRef:
          name: letsencrypt-prod-account-key
        solvers:
        - http01:
            ingress:
              ingressClassName: istio
  YAML

  # Wait for cert-manager CRDs to be available before applying
  depends_on = [helm_release.cert_manager] # Cert-manager is installed by knative.tf
}

resource "kubectl_manifest" "kagent_ui_certificate" {
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
      commonName: ${var.domain_name}
      dnsNames:
      - ${var.domain_name}
  YAML

  depends_on = [kubectl_manifest.cluster_issuer_letsencrypt]
}

resource "kubectl_manifest" "kagent_gateway" {
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
        - "${var.domain_name}"
        tls:
          httpsRedirect: true
      - port:
          number: 443
          name: https
          protocol: HTTPS
        hosts:
        - "${var.domain_name}"
        tls:
          mode: SIMPLE
          credentialName: kagent-ui-tls
  YAML

  depends_on = [helm_release.istiod]
}

resource "kubectl_manifest" "kagent_virtualservice" {
  yaml_body = <<-YAML
    apiVersion: networking.istio.io/v1alpha3
    kind: VirtualService
    metadata:
      name: kagent-ui-routing
      namespace: ${var.namespace_docs_agent}
    spec:
      hosts:
      - "${var.domain_name}"
      gateways:
      - istio-system/kagent-gateway
      http:
      - corsPolicy:
          allowOrigins:
          - regex: ".*"
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
