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

  force = true

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

  force = true

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

  force = true

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

  force = true

  depends_on = [kubectl_manifest.knative_core]
}
