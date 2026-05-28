# Namespace for Kubeflow Pipelines with Istio sidecar injection disabled
# to prevent network/proxy routing conflicts.
resource "kubernetes_namespace" "kubeflow" {
  metadata {
    name = "kubeflow"
    labels = {
      "istio-injection" = "disabled"
    }
  }
}

# Deploy Kubeflow Pipelines standalone using the official platform-agnostic dev environment manifests.
resource "null_resource" "kfp_standalone" {
  triggers = {
    version = "2.15.0"
  }

  provisioner "local-exec" {
    command = <<EOT
      kubectl apply -k github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.15.0
      kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
      kubectl apply -k github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.15.0
      kubectl set image deployment/mysql mysql=docker.io/library/mysql:8.4 -n kubeflow
      kubectl set image deployment/seaweedfs seaweedfs=docker.io/chrislusf/seaweedfs:4.00 -n kubeflow
    EOT
  }

  depends_on = [kubernetes_namespace.kubeflow]
}
