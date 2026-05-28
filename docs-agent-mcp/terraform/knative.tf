# =============================================================================
# knative.tf
# Installs Knative Serving (CRDs → Core → net-istio), cert-manager,
# Istio (base + istiod), and KServe — in dependency order.
# Post-install ConfigMap patches are applied in the same file so the
# full Knative lifecycle is visible in one place.
# =============================================================================

# --- cert-manager (required by Knative & KServe webhooks) --------------------

resource "helm_release" "cert_manager" {
  name             = "cert-manager"
  repository       = "https://charts.jetstack.io"
  chart            = "cert-manager"
  namespace        = "cert-manager"
  create_namespace = true
  version          = var.cert_manager_version

  set {
    name  = "installCRDs"
    value = "true"
  }
}

# --- Knative Serving CRDs ----------------------------------------------------

data "http" "knative_serving_crds" {
  url = "https://github.com/knative/serving/releases/download/knative-v${var.knative_version}/serving-crds.yaml"
}

data "kubectl_file_documents" "knative_crds" {
  content = data.http.knative_serving_crds.response_body
}

resource "kubectl_manifest" "knative_crds" {
  for_each  = data.kubectl_file_documents.knative_crds.manifests
  yaml_body = each.value

  depends_on = [helm_release.cert_manager]
}

# --- Knative Serving Core ----------------------------------------------------

data "http" "knative_serving_core" {
  url = "https://github.com/knative/serving/releases/download/knative-v${var.knative_version}/serving-core.yaml"
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

# --- Istio (base + istiod) ---------------------------------------------------

resource "helm_release" "istio_base" {
  name             = "istio-base"
  repository       = "https://istio-release.storage.googleapis.com/charts"
  chart            = "base"
  namespace        = "istio-system"
  create_namespace = true
  version          = var.istio_version
}

resource "helm_release" "istiod" {
  name             = "istiod"
  repository       = "https://istio-release.storage.googleapis.com/charts"
  chart            = "istiod"
  namespace        = "istio-system"
  create_namespace = true
  version          = var.istio_version

  depends_on = [helm_release.istio_base]
}

# --- Knative net-istio (networking layer) ------------------------------------

data "http" "knative_net_istio" {
  url = "https://github.com/knative/net-istio/releases/download/knative-v${var.knative_version}/net-istio.yaml"
}

data "kubectl_file_documents" "net_istio" {
  content = data.http.knative_net_istio.response_body
}

resource "kubectl_manifest" "net_istio" {
  for_each  = data.kubectl_file_documents.net_istio.manifests
  yaml_body = each.value

  depends_on = [kubectl_manifest.knative_core, helm_release.istiod]
}

# --- KServe ------------------------------------------------------------------

resource "helm_release" "kserve" {
  name             = "kserve-resources"
  repository       = "oci://ghcr.io/kserve/charts"
  chart            = "kserve-resources"
  namespace        = "kserve"
  create_namespace = true
  version          = var.kserve_version

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
    value = var.kserve_version
  }
  set {
    name  = "kserve.controller.ingress.disableIstioVirtualHost"
    value = "false"
  }

  depends_on = [kubectl_manifest.net_istio]
}

# =============================================================================
# Post-Install ConfigMap Patches
# These patch ConfigMaps that Knative Core creates. They MUST run after
# knative_core resources are established.
# =============================================================================

resource "kubernetes_config_map_v1_data" "knative_deployment_config" {
  metadata {
    name      = "config-deployment"
    namespace = "knative-serving"
  }

  data = {
    "progress-deadline"                 = "3600s"
    "max-revision-timeout-seconds"      = "3600"
    "registries-skipping-tag-resolving" = "ko.local,dev.local,docker.io,index.docker.io,quay.io,ghcr.io"
  }

  force      = true
  depends_on = [kubectl_manifest.knative_core]
}

resource "kubernetes_config_map_v1_data" "knative_features_config" {
  metadata {
    name      = "config-features"
    namespace = "knative-serving"
  }

  data = {
    "kubernetes.podspec-nodeselector" = "enabled"
    "kubernetes.podspec-tolerations"  = "enabled"
  }

  force      = true
  depends_on = [kubectl_manifest.knative_core]
}

resource "kubernetes_config_map_v1_data" "knative_autoscaler_config" {
  metadata {
    name      = "config-autoscaler"
    namespace = "knative-serving"
  }

  data = {
    "enable-scale-to-zero"               = "true"
    "scale-to-zero-pod-retention-period" = "12h"
    "scale-to-zero-grace-period"         = "30s"
  }

  force      = true
  depends_on = [kubectl_manifest.knative_core]
}

resource "kubernetes_config_map_v1_data" "knative_network_config" {
  metadata {
    name      = "config-network"
    namespace = "knative-serving"
  }

  data = {
    "ingress-class" = "istio.ingress.networking.knative.dev"
  }

  force      = true
  depends_on = [kubectl_manifest.knative_core]
}
