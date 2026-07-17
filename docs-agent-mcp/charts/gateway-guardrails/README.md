# gateway-guardrails

Helm chart for the public gateway of the Kubeflow docs-agent chatbot.
This chart is the **single source of truth** for the Istio edge config —
it replaces the raw `kubectl_manifest` heredocs that previously lived in
`terraform/istio_policies.tf` and `terraform/kagent_ingress.tf`, and the
standalone YAML under `manifests/istio*`.

## What it manages

| Layer | Resources | Toggle |
|---|---|---|
| TLS | ClusterIssuer (Let's Encrypt) + Certificate | `gateway.tls.acme.enabled` |
| Routing | Gateway + VirtualService (CORS lockdown, 30s timeout) | `gateway.enabled` |
| Rate limit L1 | `a2a-global-ratelimit` EnvoyFilter — 60 req/min token bucket on the :443 listener | `rateLimit.global.enabled` |
| Rate limit L2 | Envoy ratelimit service + Redis + EnvoyFilter — 10 req/min per client IP | `rateLimit.perIP.enabled` (off: needs real client IPs, see below) |
| Mesh policies | 8 AuthorizationPolicies for the RAG stack (Milvus, LLM, MCP, embeddings) | `meshPolicies.enabled` |

## Why

The chatbot is backed by a single GPU-served LLM. Without limits, one script can
exhaust GPU capacity (cost + DoS), and wide-open CORS let any origin embed it.
These guardrails cap request volume and lock down who may call the endpoint —
all at the Istio ingress, with no application changes.

## Install / upgrade

```bash
helm upgrade --install gateway-guardrails . -n docs-agent \
  --set domain=agent.example.com \
  --set gateway.tls.acme.email=you@example.com
```

Tuning examples:

```bash
--set rateLimit.global.requestsPerMinute=120
--set routing.cors.allowOrigins[0].exact=https://your-widget.example.app
```

## Per-IP rate limit prerequisite

`rateLimit.perIP` ships **disabled**: the ingress Service runs
`externalTrafficPolicy: Cluster`, so client IPs are SNAT'd to node IPs and a
per-IP descriptor would act as one shared bucket — effectively a second, tighter
global limit rather than per-client fairness. Before enabling, either set
`externalTrafficPolicy: Local` on the ingress Service (and verify the LB health
checks) or trust XFF via `meshConfig.numTrustedProxies`.

## Verifying

```bash
# Rate limit: expect a mix of 200/429 past the per-minute cap
seq 1 100 | xargs -P 25 -I{} curl -s -o /dev/null -w "%{http_code}\n" \
  -X POST https://$DOMAIN/api/a2a/docs-agent/kubeflow-docs-agent \
  -H 'content-type: application/json' -d '{"jsonrpc":"2.0","id":"t","method":"x"}' | sort | uniq -c

# CORS: the allowed origin gets an allow-origin header; others do not
curl -s -o /dev/null -D - -X OPTIONS https://$DOMAIN/api/a2a/docs-agent/kubeflow-docs-agent \
  -H "Origin: https://your-widget.example.app" -H "Access-Control-Request-Method: POST" \
  | grep -i access-control-allow-origin
```

## Note on the shared ingress gateway

`knative-ingress-gateway` and `knative-local-gateway` also select
`istio: ingressgateway` (the agent → LLM path rides knative-local on the same
pod). The rate-limit EnvoyFilter is therefore scoped to the `:443` listener so
it never applies to the knative / internal LLM listeners.
