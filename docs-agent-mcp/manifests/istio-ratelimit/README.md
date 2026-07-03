# A2A Gateway Guardrails (Phase 0)

Edge guardrails for the public Kubeflow Docs chatbot (A2A path), applied at the
Istio ingress gateway that fronts `agent.santhoshtoorpu.com` → `kagent-ui:8080`.

Verified against the **GSOC-2026** OKE cluster (Istio 1.23, ingress LB is the only
public front door — no `kagent-ui-lb` bypass here).

## What this protects
The chatbot is backed by a single GPU-served LLM (Qwen). Without limits, one script
can exhaust GPU capacity (cost + DoS). These guardrails cap request volume and lock
down who may call the endpoint.

## Layers

| File | Guardrail | New infra | Deploy status |
|------|-----------|-----------|---------------|
| `local-ratelimit-global.yaml` | Global cap: **60 req/min** across the ingress gateway | none | ready for demo |
| (`kagent-ui-routing` VS edit) | CORS locked to the Netlify origin + 30s route timeout | none | ready for demo |
| `perip-ratelimit-service.yaml` | `ratelimit` service + Redis, **10 req/min per IP** | Redis + ratelimit | needs prerequisites (below) |
| `perip-global-ratelimit.yaml` | Wires the gateway to the ratelimit service, keyed on client IP | none | needs prerequisites (below) |

## Prerequisites for per-IP (raise with mentors)
Per-IP limiting is a **no-op** until the real client IP reaches Envoy. Today the
ingress gateway Service is `externalTrafficPolicy: Cluster`, so kube-proxy SNATs the
source address and every request looks like it came from a node IP.

Fix one of:
- **Simplest:** set the ingress gateway Service to `externalTrafficPolicy: Local`
  (preserves L4 source IP; verify OCI LB health checks stay green), **or**
- Front with an L7 LB / PROXY protocol and set `meshConfig.numTrustedProxies` so
  Envoy trusts `X-Forwarded-For`.

Then deploy `perip-ratelimit-service.yaml` and `perip-global-ratelimit.yaml`.

## Apply / test
```bash
# validate without changing the cluster
kubectl apply --dry-run=server -f local-ratelimit-global.yaml

# demo-ready set
kubectl apply -f local-ratelimit-global.yaml
kubectl apply -f ../istio-tls/kagent-gateway-tls.yaml   # updated CORS + timeout

# verify the global cap (expect some HTTP 429 past 60/min)
for i in $(seq 1 80); do \
  curl -s -o /dev/null -w "%{http_code}\n" https://agent.santhoshtoorpu.com/ ; done | sort | uniq -c
```

## Follow-ups
- Mirror these into Terraform / Helm (ties into the CRD→Helm workstream).
- Tune `60/min` global and `10/min` per-IP against measured GPU throughput.
- Add app-layer guardrails (prompt length cap, per-session message cap) in the agent.
