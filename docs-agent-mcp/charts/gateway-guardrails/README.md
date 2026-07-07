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
| Session auth | session-issuer Deployment/Service + RequestAuthentication (+ DENY policy) | `sessionAuth.enabled` / `sessionAuth.enforce` |
| Mesh policies | 8 AuthorizationPolicies for the RAG stack (Milvus, LLM, MCP, embeddings) | `meshPolicies.enabled` |

## Session auth flow (no user accounts)

```
widget loads --> POST /api/session ------------> session-issuer mints RS256 JWT
                                                 (sub: random id, exp: +30 min)
widget calls A2A with Authorization: Bearer <jwt>
        --> Istio RequestAuthentication validates signature + expiry at the edge
        --> AuthorizationPolicy (enforce=true) rejects token-less requests
```

- `sessionAuth.enforce: false` (default) is **canary mode**: invalid/expired
  tokens are rejected, token-less requests still pass. Flip to `true` only
  after the chat widget attaches tokens, or the public demo breaks.
- Token invalidation = expiry (`exp`). No revocation store needed.
- Prerequisite: signing-key Secret (terraform creates it automatically):

  ```bash
  openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out private.pem
  kubectl -n docs-agent create secret generic session-issuer-key --from-file=private.pem
  ```

## Install / upgrade

```bash
helm upgrade --install gateway-guardrails . -n docs-agent \
  --set domain=agent.example.com \
  --set gateway.tls.acme.email=you@example.com
```

Tuning examples:

```bash
--set rateLimit.global.requestsPerMinute=120
--set sessionAuth.enforce=true
--set routing.cors.allowOrigins[0].exact=https://your-widget.example.app
```

## Per-IP rate limit prerequisite

`rateLimit.perIP` ships **disabled**: the ingress Service runs
`externalTrafficPolicy: Cluster`, so client IPs are SNAT'd to node IPs and a
per-IP descriptor would act as one shared bucket. Before enabling, either set
`externalTrafficPolicy: Local` on the ingress Service (verify LB health
checks) or trust XFF via `meshConfig.numTrustedProxies`.

## Verifying

```bash
# Rate limit: expect a mix of 200/429 past the per-minute cap
seq 1 100 | xargs -P 25 -I{} curl -s -o /dev/null -w "%{http_code}\n" \
  -X POST https://$DOMAIN/api/a2a/docs-agent/kubeflow-docs-agent \
  -H 'content-type: application/json' -d '{"jsonrpc":"2.0","id":"t","method":"x"}' | sort | uniq -c

# Session: mint a token, then call A2A with it
TOKEN=$(curl -s -X POST https://$DOMAIN/api/session | jq -r .access_token)
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
  https://$DOMAIN/api/a2a/docs-agent/kubeflow-docs-agent \
  -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":"t","method":"x"}'

# A garbage token must be rejected (401) even with enforce=false
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
  https://$DOMAIN/api/a2a/docs-agent/kubeflow-docs-agent \
  -H "authorization: Bearer garbage" -H 'content-type: application/json' -d '{}'
```
