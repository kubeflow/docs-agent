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
| Routing | Gateway + VirtualService: only `/api/session`, the JWKS and the A2A stream are public (CORS lockdown, 300s stream timeout) | `gateway.enabled` |
| Rate limit L1 | `a2a-global-ratelimit` EnvoyFilter — 60 req/min token bucket on the :443 listener | `rateLimit.global.enabled` |
| Rate limit L2 | Envoy ratelimit service + Redis + EnvoyFilter — 10 req/min per client IP | `rateLimit.perIP.enabled` (off: needs real client IPs, see below) |
| Mesh policies | AuthorizationPolicies for the RAG stack (Milvus, MCP, embeddings, plus one per `meshPolicies.llmInferenceServiceNames` entry) | `meshPolicies.enabled` |

## Why

The chatbot is backed by a single GPU-served LLM. Without limits, one script can
exhaust GPU capacity (cost + DoS), and wide-open CORS let any origin embed it.
These guardrails cap request volume and lock down who may call the endpoint —
all at the Istio ingress, with no application changes.

## Install / upgrade

The ACME HTTP-01 solver needs the `istio` IngressClass. Apply it once per
cluster: `kubectl apply -f ../../manifests/istio-tls/ingress-class.yaml`.
`domain` and `gateway.tls.acme.email` have no defaults. Terraform sets them from
`kagent_domain_name` and `kagent_acme_email`; for a manual upgrade, pass them:

```bash
helm upgrade --install gateway-guardrails . -n docs-agent \
  --set domain=agent.your-domain.example \
  --set gateway.tls.acme.email=you@your-domain.example
```

Session tokens are enforced (`sessionAuth.enforce: true`) on `/a2a/*` and
`/api/a2a/*`: a request without a valid token from `POST /api/session` gets a 403.

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
# Exposure: kagent's admin API (no auth) must not be reachable; expect 404 for all
for p in / /api/agents /api/sessions /api/toolservers '/a2a/..%2fapi/sessions'; do
  curl -s -o /dev/null -w "%{http_code} $p\n" --path-as-is "https://$DOMAIN$p"; done

# Session auth: no token -> 403
curl -s -o /dev/null -w "%{http_code}\n" -X POST https://$DOMAIN/a2a/docs-agent/kubeflow-docs-agent \
  -H 'content-type: application/json' -d '{}'

# CORS: a foreign origin must get no access-control-allow-origin header
TOKEN=$(curl -s -X POST https://$DOMAIN/api/session | jq -r .access_token)
curl -s -o /dev/null -D - https://$DOMAIN/api/a2a/docs-agent/kubeflow-docs-agent/.well-known/agent.json \
  -H "Authorization: Bearer $TOKEN" -H "Origin: https://evil.example" | grep -i access-control-allow-origin

# Rate limit: expect 429s past the per-minute cap (403 for the rest: no token)
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
