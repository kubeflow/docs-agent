#!/usr/bin/env bash
# Bring up Qwen2.5-3B-Instruct on KServe (OKE GPU node).
#
# Prerequisites:
#   - GPU node pool with nvidia.com/gpu label/taint
#   - KServe installed (Knative mode)
#   - kubectl context pointing at OKE cluster
#
# Usage:
#   ./scripts/deploy-qwen-kserve.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAMESPACE="${NAMESPACE:-docs-agent}"

echo "==> Ensuring GPU node root volume is expanded (250GB boot volume -> ~239GB root FS)"
kubectl apply -f "${ROOT_DIR}/legacy/manifests/gpu-node-lvm-expand-job.yaml"
if kubectl wait --for=condition=complete "job/gpu-node-lvm-expand" -n kube-system --timeout=180s 2>/dev/null; then
  kubectl logs -n kube-system "job/gpu-node-lvm-expand" || true
else
  echo "WARN: LVM expand job did not complete in time; continuing if GPU node already expanded."
fi

if [[ -n "${HF_TOKEN:-${hf_token:-}}" ]]; then
  TOKEN="${HF_TOKEN:-${hf_token:-}}"
  echo "==> Applying huggingface-secret"
  kubectl create secret generic huggingface-secret -n "${NAMESPACE}" \
    --from-literal=token="${TOKEN}" \
    --dry-run=client -o yaml | kubectl apply -f -
else
  echo "==> HF_TOKEN not set; skipping huggingface-secret (Qwen2.5-3B-Instruct is public)"
fi

echo "==> Applying KServe ServingRuntime + InferenceService (single-GPU recycle)"
FORCE_KSERVE_DEPLOY=true "${ROOT_DIR}/scripts/deploy-kserve-if-changed.sh"

URL="$(kubectl get inferenceservice qwen -n "${NAMESPACE}" -o jsonpath='{.status.url}' 2>/dev/null || true)"
PREDICTOR_HOST="qwen-llm.${NAMESPACE}.svc.cluster.local"
echo "==> InferenceService Ready"
echo "    status URL: ${URL:-n/a}"
echo "    in-cluster OpenAI base (Kagent): http://${PREDICTOR_HOST}/openai/v1"

echo "==> Smoke test (chat completion via KServe OpenAI endpoint)"
POD="$(kubectl get pods -n "${NAMESPACE}" -l serving.kserve.io/inferenceservice=qwen -o jsonpath='{.items[0].metadata.name}')"
kubectl exec -n "${NAMESPACE}" "${POD}" -c kserve-container -- python3 -c "
import urllib.request, json
host='http://127.0.0.1:8080/openai/v1/chat/completions'
payload=json.dumps({'model':'qwen2.5-3b-instruct','messages':[{'role':'user','content':'Say hello in one sentence.'}],'max_tokens':32}).encode()
req=urllib.request.Request(host, data=payload, headers={'Content-Type':'application/json'})
print(urllib.request.urlopen(req, timeout=120).read().decode())
" || echo "Smoke test failed — see docs/QWEN-OKE-BRINGUP.md"
