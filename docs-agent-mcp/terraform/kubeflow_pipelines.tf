# =============================================================================
# kubeflow_pipelines.tf
# Deploys Kubeflow Pipelines (and optional Kubeflow platform components) using
# the new experimental Kubeflow Helm charts from `kubeflow/manifests`
# (`deployments/charts/kubeflow-manifests`), with fallback to standalone
# Kustomize when `var.use_kubeflow_helm_charts = false`.
# =============================================================================

locals {
  kf_charts_root = "${path.module}/../../deployments/charts/kubeflow-manifests"
}

resource "kubernetes_namespace" "kubeflow" {
  metadata {
    name = var.namespace_kubeflow
    labels = {
      "control-plane"                        = "kubeflow"
      "istio-injection"                      = "disabled"
      "pod-security.kubernetes.io/enforce"   = "baseline"
      "pod-security.kubernetes.io/warn"      = "restricted"
    }
  }
}

# Self-signed ClusterIssuer required by Kubeflow Helm charts (e.g. kfp-cache-cert)
resource "kubectl_manifest" "kubeflow_self_signing_issuer" {
  count = var.use_kubeflow_helm_charts ? 1 : 0

  yaml_body = <<YAML
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: kubeflow-self-signing-issuer
  labels:
    app.kubernetes.io/component: cert-manager
    app.kubernetes.io/name: cert-manager
spec:
  selfSigned: {}
YAML

  depends_on = [helm_release.cert_manager]
}

# Kubeflow aggregated RBAC ClusterRoles chart (`common/kubeflow-roles/helm`)
resource "helm_release" "kubeflow_roles" {
  count            = var.use_kubeflow_helm_charts ? 1 : 0
  name             = "kubeflow-roles"
  chart            = "${local.kf_charts_root}/common/kubeflow-roles/helm"
  namespace        = "default"
  create_namespace = false

  depends_on = [kubernetes_namespace.kubeflow]
}

# New Experimental Kubeflow Pipelines Helm Chart (`applications/pipeline/helm`)
# Phase 1: Install CRDs first so CustomResourceDefinitions are Established before
# the controller and Application/DecoratorController CRs are created.
resource "helm_release" "kubeflow_pipelines" {
  count            = var.use_kubeflow_helm_charts ? 1 : 0
  name             = "kubeflow-pipelines"
  chart            = "${local.kf_charts_root}/applications/pipeline/helm"
  namespace        = var.namespace_kubeflow
  create_namespace = false
  timeout          = 600

  set {
    name  = "scenario"
    value = var.kubeflow_pipelines_scenario
  }
  set {
    name  = "crds.enabled"
    value = "true"
  }
  set {
    name  = "install.enabled"
    value = "false"
  }

  depends_on = [
    kubernetes_namespace.kubeflow,
    kubectl_manifest.kubeflow_self_signing_issuer,
    helm_release.kubeflow_roles,
  ]
}

# Phase 2: Wait for Pipelines CRDs to establish, then upgrade the Helm release
# with `install.enabled = true` (matches upstream `tests/pipelines_helm_install.sh`).
resource "null_resource" "kubeflow_pipelines_workload_upgrade" {
  count = var.use_kubeflow_helm_charts ? 1 : 0

  triggers = {
    scenario = var.kubeflow_pipelines_scenario
    release  = helm_release.kubeflow_pipelines[0].id
  }

  provisioner "local-exec" {
    interpreter = ["/bin/sh", "-c"]
    environment = {
      KUBECONFIG = local.kubeconfig
    }
    command = <<-EOT
      set -e
      for crd in $(helm get manifest kubeflow-pipelines -n ${var.namespace_kubeflow} | awk '$0 == "kind: CustomResourceDefinition" { crd=1; next } crd && /^  name: / { print $2; crd=0 }'); do
        kubectl wait --for=condition=Established "crd/$crd" --timeout=120s
      done
      helm upgrade kubeflow-pipelines "${local.kf_charts_root}/applications/pipeline/helm" \
        --namespace ${var.namespace_kubeflow} \
        --values "${local.kf_charts_root}/applications/pipeline/helm/ci/values-${var.kubeflow_pipelines_scenario}.yaml" \
        --wait --timeout 15m
    EOT
  }

  depends_on = [helm_release.kubeflow_pipelines]
}

# Optional Additional Experimental Kubeflow Platform Helm Charts
resource "helm_release" "kubeflow_hub" {
  count            = var.use_kubeflow_helm_charts && var.enable_kubeflow_full_platform_helm ? 1 : 0
  name             = "kubeflow-hub"
  chart            = "${local.kf_charts_root}/experimental/helm/charts/hub"
  namespace        = var.namespace_kubeflow
  create_namespace = false

  depends_on = [kubernetes_namespace.kubeflow, helm_release.istiod]
}

resource "helm_release" "kserve_ui" {
  count            = var.use_kubeflow_helm_charts && var.enable_kubeflow_full_platform_helm ? 1 : 0
  name             = "kserve-ui"
  chart            = "${local.kf_charts_root}/experimental/helm/charts/kserve-ui"
  namespace        = "kserve"
  create_namespace = false

  depends_on = [helm_release.kserve]
}

# Legacy Kustomize fallback when var.use_kubeflow_helm_charts == false
resource "null_resource" "kfp_standalone" {
  count = var.use_kubeflow_helm_charts ? 0 : 1

  triggers = {
    version = var.kfp_version
  }

  provisioner "local-exec" {
    interpreter = ["/bin/sh", "-c"]
    environment = {
      KUBECONFIG = local.kubeconfig
    }
    command = <<-EOT
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
