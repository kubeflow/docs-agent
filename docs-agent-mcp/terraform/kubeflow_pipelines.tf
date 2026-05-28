# =============================================================================
# kubeflow_pipelines.tf
# Deploys Kubeflow Pipelines standalone in its own namespace with Istio
# sidecar injection disabled to avoid mesh routing conflicts.
#
# Uses null_resource + local-exec because KFP ships kustomize-only manifests
# with no official Helm chart. The interpreter is set explicitly so the same
# script works on Linux CI runners and Windows (Git Bash / WSL).
# =============================================================================

resource "kubernetes_namespace" "kubeflow" {
  metadata {
    name = var.namespace_kubeflow
    labels = {
      # Disable Istio sidecar injection — KFP has its own internal networking
      # and injecting sidecars causes routing conflicts with Argo Workflows.
      "istio-injection" = "disabled"
    }
  }
}

resource "null_resource" "kfp_standalone" {
  triggers = {
    version = var.kfp_version
  }

  provisioner "local-exec" {
    # Use /bin/sh explicitly so this works on Linux CI runners.
    # On Windows, run via Git Bash, WSL, or a Linux CI agent.
    interpreter = ["/bin/sh", "-c"]
    command     = <<-EOT
      set -e
      kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${var.kfp_version}"
      kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
      kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${var.kfp_version}"
      kubectl set image deployment/mysql mysql=docker.io/library/mysql:8.4 -n ${var.namespace_kubeflow}
      kubectl set image deployment/seaweedfs seaweedfs=docker.io/chrislusf/seaweedfs:4.00 -n ${var.namespace_kubeflow}
      kubectl patch deployment seaweedfs -n ${var.namespace_kubeflow} --type=json \
        -p='[{"op":"add","path":"/spec/template/spec/securityContext/fsGroup","value":1000}]'
    EOT
  }

  depends_on = [kubernetes_namespace.kubeflow]
}
