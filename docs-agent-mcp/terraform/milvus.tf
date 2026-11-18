# =============================================================================
# milvus.tf
# Deploys the Milvus Operator (via Helm), a lightweight SeaweedFS S3-compatible
# object store (replacing MinIO, matching Kubeflow 2.18's Apache-2.0 default on
# GHCR), and a standalone Milvus instance (via Milvus CRD) in ml-infra.
#
# Using `storage.external = true` backed by SeaweedFS in dev means switching to
# OCI Object Storage S3 API or AWS S3 in prod only changes the endpoint/Secret.
# =============================================================================

variable "seaweedfs_image" {
  description = "Apache-2.0 SeaweedFS image hosted on GHCR (matches Kubeflow Pipelines default)"
  type        = string
  default     = "ghcr.io/chrislusf/seaweedfs:4.51"
}

variable "milvus_s3_access_key" {
  description = "S3 access key for Milvus object storage (SeaweedFS in dev, OCI Object Storage in prod)"
  type        = string
  default     = "milvusadmin"
}

variable "milvus_s3_secret_key" {
  description = "S3 secret key for Milvus object storage (SeaweedFS in dev, OCI Object Storage in prod)"
  type        = string
  sensitive   = true
  default     = "milvusadminsecret"
}

variable "milvus_s3_bucket" {
  description = "S3 bucket name used by Milvus for vector segments"
  type        = string
  default     = "milvus-bucket"
}

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

resource "null_resource" "wait_for_milvus_crd" {
  provisioner "local-exec" {
    interpreter = ["/bin/sh", "-c"]
    environment = {
      KUBECONFIG = local.kubeconfig
    }
    command = "kubectl wait --for condition=established --timeout=120s crd/milvuses.milvus.io"
  }

  depends_on = [helm_release.milvus_operator]
}

# --- SeaweedFS S3 Object Storage for Milvus (No MinIO) -----------------------

resource "kubernetes_secret" "milvus_seaweedfs" {
  metadata {
    name      = "milvus-seaweedfs-secret"
    namespace = var.namespace_ml_infra
    labels = {
      app = "seaweedfs"
    }
  }

  data = {
    accesskey = var.milvus_s3_access_key
    secretkey = var.milvus_s3_secret_key
  }

  depends_on = [kubernetes_namespace.ml_infra]
}

resource "kubectl_manifest" "milvus_seaweedfs_pvc" {
  yaml_body = <<YAML
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: milvus-seaweedfs-pvc
  namespace: ${var.namespace_ml_infra}
  labels:
    app: seaweedfs
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
YAML

  depends_on = [kubernetes_namespace.ml_infra]
}

resource "kubectl_manifest" "milvus_seaweedfs_deployment" {
  yaml_body = <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: seaweedfs
  namespace: ${var.namespace_ml_infra}
  labels:
    app: seaweedfs
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: seaweedfs
  template:
    metadata:
      labels:
        app: seaweedfs
    spec:
      securityContext:
        fsGroup: 1000
        fsGroupChangePolicy: OnRootMismatch
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: seaweedfs
          image: ${var.seaweedfs_image}
          args:
            - server
            - -dir=/data
            - -s3
            - -s3.port=8333
            - -iam
            - -filer
            - -master.volumePreallocate=false
          envFrom:
            - secretRef:
                name: milvus-seaweedfs-secret
          ports:
            - containerPort: 8333
              name: s3
            - containerPort: 9333
              name: master
            - containerPort: 8888
              name: filer
          lifecycle:
            postStart:
              exec:
                command:
                  - /bin/sh
                  - -c
                  - |
                    for i in $(seq 1 60); do
                      if wget -q --spider http://127.0.0.1:8333/status; then break; fi
                      sleep 2
                    done
                    echo "s3.bucket.create --name ${var.milvus_s3_bucket}" | /usr/bin/weed shell 2>&1 || true
                    echo "s3.configure -user milvus-admin -access_key $accesskey -secret_key $secretkey -actions Admin -apply" | /usr/bin/weed shell 2>&1 || true
          readinessProbe:
            httpGet:
              path: /status
              port: 8333
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /status
              port: 8333
            initialDelaySeconds: 30
            periodSeconds: 30
          resources:
            requests:
              cpu: 64m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 1Gi
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            runAsUser: 1000
            runAsGroup: 1000
            runAsNonRoot: true
          volumeMounts:
            - name: data
              mountPath: /data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: milvus-seaweedfs-pvc
YAML

  depends_on = [
    kubernetes_secret.milvus_seaweedfs,
    kubectl_manifest.milvus_seaweedfs_pvc,
  ]
}

resource "kubectl_manifest" "milvus_seaweedfs_service" {
  yaml_body = <<YAML
apiVersion: v1
kind: Service
metadata:
  name: seaweedfs
  namespace: ${var.namespace_ml_infra}
  labels:
    app: seaweedfs
spec:
  selector:
    app: seaweedfs
  ports:
    - name: s3
      port: 8333
      targetPort: 8333
      protocol: TCP
    - name: s3-compat
      port: 9000
      targetPort: 8333
      protocol: TCP
YAML

  depends_on = [kubectl_manifest.milvus_seaweedfs_deployment]
}

# --- Standalone Milvus CR backed by SeaweedFS S3 -----------------------------

resource "kubectl_manifest" "milvus_standalone" {
  yaml_body = <<YAML
apiVersion: milvus.io/v1beta1
kind: Milvus
metadata:
  name: milvus
  namespace: ${var.namespace_ml_infra}
spec:
  mode: standalone
  config:
    minio:
      bucketName: ${var.milvus_s3_bucket}
      useSSL: false
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
      external: true
      type: S3
      endpoint: seaweedfs.${var.namespace_ml_infra}.svc.cluster.local:8333
      secretRef: ${kubernetes_secret.milvus_seaweedfs.metadata[0].name}
  components:
    image: docker.io/milvusdb/milvus:${var.milvus_version}
    resources:
      requests:
        cpu: 500m
        memory: 2Gi
      limits:
        cpu: 2
        memory: 8Gi
YAML

  depends_on = [
    null_resource.wait_for_milvus_crd,
    kubernetes_namespace.ml_infra,
    kubectl_manifest.milvus_seaweedfs_service,
  ]
}
