# =============================================================================
# milvus.tf
# Deploys the Milvus Operator (via Helm) and a lightweight standalone
# Milvus instance (via Milvus CRD) in the ml-infra namespace.
#
# The operator pattern means Milvus lifecycle (upgrades, failover) is managed
# by the operator controller — Terraform only owns the CR spec.
# =============================================================================

resource "helm_release" "milvus_operator" {
  name             = "milvus-operator"
  repository       = "https://zilliztech.github.io/milvus-operator"
  chart            = "milvus-operator"
  namespace        = "milvus-operator"
  create_namespace = true

  # OKE enforces short-name registry policy — must use FQDN to avoid
  # "ambiguous image name" ImageInspectError on node pull.
  set {
    name  = "image.repository"
    value = "docker.io/milvusdb/milvus-operator"
  }
}

# Wait for the Milvus CRD to be registered before creating a Milvus CR.
# Uses /bin/sh explicitly for cross-platform CI compatibility.
resource "null_resource" "wait_for_milvus_crd" {
  provisioner "local-exec" {
    interpreter = ["/bin/sh", "-c"]
    command     = "kubectl wait --for condition=established --timeout=120s crd/milvuses.milvus.io"
  }

  depends_on = [helm_release.milvus_operator]
}

# Lightweight standalone Milvus CR — all resource limits set low intentionally
# to keep this on the CPU nodes and leave GPU nodes free for inference.
resource "kubectl_manifest" "milvus_standalone" {
  yaml_body = <<YAML
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: milvus
  namespace: ${var.namespace_ml_infra}
spec:
  mode: standalone
  dependencies:
    etcd:
      inCluster:
        values:
          image:
            # Note: no docker.io/ prefix here — the operator's internal etcd
            # Helm chart prepends docker.io/ itself. Adding it again causes
            # a double-prefix pull failure (docker.io/docker.io/...).
            repository: milvusdb/etcd
            tag: v3.5.5-r4
          resources:
            requests:
              cpu: 50m
              memory: 128Mi
            limits:
              cpu: 200m
              memory: 256Mi
    storage:
      inCluster:
        values:
          image:
            repository: docker.io/minio/minio
            tag: RELEASE.2023-03-20T20-16-18Z
          resources:
            requests:
              cpu: 50m
              memory: 128Mi
            limits:
              cpu: 200m
              memory: 256Mi
  components:
    image: docker.io/milvusdb/milvus:${var.milvus_version}
    resources:
      requests:
        cpu: 100m
        memory: 256Mi
      limits:
        cpu: 500m
        memory: 512Mi
YAML

  depends_on = [null_resource.wait_for_milvus_crd, kubernetes_namespace.ml_infra]
}
