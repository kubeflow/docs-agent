# Cert-Manager (Required by Knative and KServe webhooks)
resource "helm_release" "cert_manager" {
  name             = "cert-manager"
  repository       = "https://charts.jetstack.io"
  chart            = "cert-manager"
  namespace        = "cert-manager"
  create_namespace = true
  version          = "v1.20.2"


  set {
    name  = "installCRDs"
    value = "true"
  }
}

# Knative Serving CRDs
data "http" "knative_serving_crds" {
  url = "https://github.com/knative/serving/releases/download/knative-v1.22.0/serving-crds.yaml"
}

data "kubectl_file_documents" "knative_crds" {
  content = data.http.knative_serving_crds.response_body
}

resource "kubectl_manifest" "knative_crds" {
  for_each  = data.kubectl_file_documents.knative_crds.manifests
  yaml_body = each.value

  depends_on = [helm_release.cert_manager]
}

# Knative Serving Core
data "http" "knative_serving_core" {
  url = "https://github.com/knative/serving/releases/download/knative-v1.22.0/serving-core.yaml"
}

data "kubectl_file_documents" "knative_core" {
  content = data.http.knative_serving_core.response_body
}

resource "kubectl_manifest" "knative_serving_namespace" {
  yaml_body = <<YAML
apiVersion: v1
kind: Namespace
metadata:
  name: knative-serving
YAML
}

resource "kubectl_manifest" "knative_core" {
  for_each  = data.kubectl_file_documents.knative_core.manifests
  yaml_body = each.value

  depends_on = [kubectl_manifest.knative_crds, kubectl_manifest.knative_serving_namespace]
}

# Istio Base
resource "helm_release" "istio_base" {
  name             = "istio-base"
  repository       = "https://istio-release.storage.googleapis.com/charts"
  chart            = "base"
  namespace        = "istio-system"
  create_namespace = true
  version          = "1.23.0"
}

# Istiod
resource "helm_release" "istiod" {
  name             = "istiod"
  repository       = "https://istio-release.storage.googleapis.com/charts"
  chart            = "istiod"
  namespace        = "istio-system"
  create_namespace = true
  version          = "1.23.0"

  depends_on = [helm_release.istio_base]
}

# Knative Istio Networking Layer
data "http" "knative_net_istio" {
  url = "https://github.com/knative/net-istio/releases/download/knative-v1.22.0/net-istio.yaml"
}

data "kubectl_file_documents" "net_istio" {
  content = data.http.knative_net_istio.response_body
}

resource "kubectl_manifest" "net_istio" {
  for_each  = data.kubectl_file_documents.net_istio.manifests
  yaml_body = each.value

  depends_on = [kubectl_manifest.knative_core, helm_release.istiod]
}


# KServe Installation
resource "helm_release" "kserve" {
  name             = "kserve-resources"
  repository       = "oci://ghcr.io/kserve/charts"
  chart            = "kserve-resources"
  namespace        = "kserve"
  create_namespace = true
  version          = "v0.17.0"

  set {
    name  = "kserve.controller.deploymentMode"
    value = "Knative"
  }
  set {
    name  = "kserve.controller.image"
    value = "docker.io/kserve/kserve-controller"
  }
  set {
    name  = "kserve.controller.tag"
    value = "v0.17.0"
  }
  set {
    name  = "kserve.controller.ingress.disableIstioVirtualHost"
    value = "false"
  }

  depends_on = [kubectl_manifest.net_istio]
}
