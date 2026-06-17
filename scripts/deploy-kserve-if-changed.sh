#!/usr/bin/env bash
# Apply KServe Qwen manifests only when they change (or FORCE_KSERVE_DEPLOY=true).
# On a single-GPU node: stop the ISVC, delete old Knative revisions, then apply.
#
# Usage:
#   ./scripts/deploy-kserve-if-changed.sh              # apply only if manifest diff
#   FORCE_KSERVE_DEPLOY=true ./scripts/deploy-kserve-if-changed.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAMESPACE="${NAMESPACE:-docs-agent}"
FORCE="${FORCE_KSERVE_DEPLOY:-false}"

SR="${ROOT_DIR}/legacy/manifests/serving-runtime.yaml"
ISVC="${ROOT_DIR}/legacy/manifests/inference-service.yaml"
QWEN_SVC="${ROOT_DIR}/legacy/manifests/qwen-llm-service.yaml"

if [[ ! -f "$SR" || ! -f "$ISVC" ]]; then
  echo "ERROR: missing KServe manifests under legacy/manifests/"
  exit 1
fi

# Always safe to reconcile the stable ClusterIP service (no GPU).
kubectl apply -f "$QWEN_SVC"

if ! kubectl get inferenceservice qwen -n "$NAMESPACE" &>/dev/null; then
  echo "==> InferenceService qwen not found — first-time GPU deploy"
  FORCE=true
fi

if [[ "$FORCE" != "true" ]]; then
  # kubectl diff compares desired YAML to cluster state (ignores repo path moves).
  if kubectl diff -f "$SR" -f "$ISVC" -n "$NAMESPACE" >/dev/null 2>&1; then
    echo "==> KServe manifests unchanged on cluster — skipping GPU inference deploy"
    exit 0
  fi
  echo "==> KServe cluster diff detected — recycling GPU revision"
fi

recycle_gpu_predictor() {
  echo "==> Stopping InferenceService to free GPU"
  kubectl annotate inferenceservice qwen -n "$NAMESPACE" \
    serving.kserve.io/stop=true --overwrite

  echo "==> Waiting for qwen predictor pods to terminate"
  kubectl wait --for=delete pod -l serving.kserve.io/inferenceservice=qwen \
    -n "$NAMESPACE" --timeout=600s 2>/dev/null || true

  echo "==> Deleting stale Knative revisions and predictor deployments"
  kubectl delete revisions -n "$NAMESPACE" \
    -l serving.knative.dev/service=qwen-predictor \
    --ignore-not-found --wait=true 2>/dev/null || true
  kubectl delete deploy -n "$NAMESPACE" \
    -l serving.kserve.io/inferenceservice=qwen \
    --ignore-not-found --wait=true 2>/dev/null || true

  # Brief pause so the GPU device plugin releases the card
  sleep 15
}

recycle_gpu_predictor

echo "==> Applying ServingRuntime + InferenceService"
kubectl apply -f "$SR"
kubectl apply -f "$ISVC"

echo "==> Starting InferenceService"
kubectl annotate inferenceservice qwen -n "$NAMESPACE" \
  serving.kserve.io/stop- --overwrite 2>/dev/null || true

echo "==> Waiting for InferenceService Ready (up to 30m)"
kubectl wait --for=condition=Ready inferenceservice/qwen \
  -n "$NAMESPACE" --timeout=1800s

echo "==> KServe deploy complete"
kubectl get inferenceservice qwen -n "$NAMESPACE" \
  -o jsonpath='{.status.components.predictor.latestReadyRevision}{"\n"}'
