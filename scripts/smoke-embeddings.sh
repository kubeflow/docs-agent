#!/usr/bin/env bash
# Smoke-test TEI embeddings reachable from the MCP pod (ml-infra stack).
set -euo pipefail

NAMESPACE="${DOCS_AGENT_NS:-docs-agent}"
DEPLOY="${MCP_DEPLOYMENT:-mcp-kubeflow-docs}"

kubectl rollout status "deployment/${DEPLOY}" -n "${NAMESPACE}" --timeout=120s

kubectl exec -n "${NAMESPACE}" "deploy/${DEPLOY}" -- python3 -c "
import os, requests
url = os.environ.get('EMBEDDINGS_URL', '').strip()
if not url:
    raise SystemExit('EMBEDDINGS_URL not set on MCP deployment')
texts = ['kubeflow pipelines', 'KServe InferenceService']
r = requests.post(url, json={'inputs': texts}, timeout=120)
r.raise_for_status()
vecs = r.json()
assert len(vecs) == len(texts), (len(vecs), len(texts))
dim = len(vecs[0])
assert dim == 768, f'expected 768-dim vectors, got {dim}'
print(f'OK: {url} batch={len(texts)} dim={dim}')
"
