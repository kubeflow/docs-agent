# Helm Release for Milvus Operator
resource "helm_release" "milvus_operator" {
  name             = "milvus-operator"
  repository       = "https://zilliztech.github.io/milvus-operator"
  chart            = "milvus-operator"
  namespace        = "milvus-operator"
  create_namespace = true

  set {
    name  = "image.repository"
    value = "docker.io/milvusdb/milvus-operator"
  }
}

# Wait for Milvus CRD to be registered by the operator
resource "null_resource" "wait_for_milvus_crd" {
  provisioner "local-exec" {
    command = "kubectl wait --for condition=established --timeout=120s crd/milvuses.milvus.io"
  }

  depends_on = [helm_release.milvus_operator]
}

# Deploy Standalone Milvus Custom Resource in ml-infra namespace
resource "kubectl_manifest" "milvus_standalone" {
  yaml_body = <<YAML
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: milvus
  namespace: ml-infra
spec:
  mode: standalone
  dependencies:
    etcd:
      inCluster:
        values:
          image:
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
    image: docker.io/milvusdb/milvus:v2.4.15
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
