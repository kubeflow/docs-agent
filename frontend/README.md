# Kubeflow website chatbot (Kagent A2A)

Static assets for the docs-site chat widget. The bot talks to **Kagent** over JSON-RPC `message/stream` (not the legacy REST API).

## Configure the agent URL

Set one of these before the widget loads:

```html
<script>
  window.KUBEFLOW_DOCS_AGENT_URL = 'https://YOUR_HOST/a2a/docs-agent/kubeflow-docs-agent';
</script>
<script src="/docs_scripts/chatbot.js" defer></script>
```

Or on the script tag:

```html
<script
  src="/docs_scripts/chatbot.js"
  data-agent-url="https://YOUR_HOST/a2a/docs-agent/kubeflow-docs-agent"
  defer
></script>
```

Or base URL only (path is appended automatically):

```html
<script
  src="/docs_scripts/chatbot.js"
  data-agent-base="http://YOUR_LOAD_BALANCER_IP"
  defer
></script>
```

## Vercel demo page

Host `docs_scripts/` and `docs_styles/` on Vercel and set `KUBEFLOW_DOCS_AGENT_URL` to your cluster endpoint:

- **LoadBalancer (current OKE setup):** `kagent-ui-lb` external IP + `/a2a/docs-agent/kubeflow-docs-agent` (use HTTPS ingress if the Vercel site is `https://` to avoid mixed-content blocking).
- **Istio ingress (optional Terraform):** set `enable_kagent_ingress = true` and `kagent_domain_name` in `docs-agent-mcp/terraform/`, then point DNS and use `https://your-domain/...`.

CORS: browser calls require the agent host to allow your site's origin. The Istio ingress only allows the exact origins listed in `kagent_cors_allow_origins` (terraform) / `routing.cors.allowOrigins` (gateway-guardrails Helm chart) — add your deployment's origin there; the LB path depends on Kagent's own CORS settings.

Session auth: when `sessionAuth.enforce=true` is set on the gateway-guardrails chart, the widget must first `POST /api/session` to get a short-lived bearer token and send it as `Authorization: Bearer <token>` on every A2A call (re-fetch on 401).

## Milvus collections (MCP tools)

| Tool | Collection |
|------|------------|
| Docs | `kubeflow_docs` |
| Issues | `issues_rag` |
| Code | `code_rag` |
