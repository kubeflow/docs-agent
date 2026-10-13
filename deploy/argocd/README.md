# Argo CD (observe-only)

Argo CD watches the Helm releases (`docs-agent`, `gateway-guardrails`,
`llm-runtime`, `llm-judge`, `langfuse`) against `kubeflow/docs-agent` `main`. None of the
Applications has a `syncPolicy`: Argo reports drift, and a maintainer clicks
Sync to deploy. GPU releases (`llm-runtime`, `llm-judge`) stay manual.

## What lives where

| Kind of data | Where | Who can change it |
|---|---|---|
| Charts, Argo Applications | `docs-agent-mcp/charts/`, `deploy/argocd/` | PR + review |
| Per-environment values (domain, CORS origins, image overrides, ingress hosts) | `deploy/envs/<env>/` in this repo, read by Argo as a second source (`$values/deploy/envs/prod/<app>.yaml`) | PR + review |
| Secrets (API keys, Langfuse keys, passwords, pull secrets) | Kubernetes Secrets only, referenced by name (list below) | cluster admins |

Values files are public, so nothing personal or secret goes in them. The ACME
contact email, for example, lives only in the cluster's ClusterIssuer
(`gateway.tls.acme.createIssuer: false`). Argo CD cannot read Helm values from
a Kubernetes Secret on its own, so secret values are never Helm values: the
chart takes a Secret name.

### Secrets to create before a first install

Created by a cluster admin, never committed. TLS certificates, service-account
tokens, Helm release records and the Istio CA are generated automatically and
are not listed.

| Namespace | Secret | Used for |
|---|---|---|
| `docs-agent` | `kagent-llm-key` | API key kagent sends to the model endpoint |
| `docs-agent` | `kagent-postgresql` | kagent database password |
| `docs-agent` | `mcp-server-secret` | MCP server settings (Milvus, GitHub) |
| `docs-agent` | `mcp-langfuse-keys` | Langfuse project keys for MCP tracing |
| `docs-agent` | `langfuse-otlp` | Langfuse keys for the OTel collector |
| `docs-agent` | `session-issuer-key` | signing key for chat session tokens |
| `docs-agent` | `ghcr-session-issuer` | pull secret for the session-issuer image |
| `langfuse` | `langfuse-app`, `langfuse-postgresql-auth`, `langfuse-clickhouse-auth`, `langfuse-redis-auth`, `langfuse-s3-auth` | Langfuse app secrets and its data stores |
| `ml-infra` | `milvus-minio` | Milvus object store |
| `kubeflow` | `milvus-auth`, `github-pat` | ingestion pipelines |
| `argocd` | `argocd-secret` | Argo CD admin password hash and server key |

## Install / upgrade

```bash
helm repo add argo https://argoproj.github.io/argo-helm
helm upgrade --install argocd argo/argo-cd -n argocd --create-namespace --version 10.10.1 \
  -f deploy/argocd/install-values.yaml
kubectl apply -f deploy/envs/prod/ingress/        # public HTTPS for Langfuse and Argo (needs A records)
kubectl apply -f deploy/platform/llm-judge/authz.yaml
kubectl apply -f deploy/argocd/apps/
```

## Admin and read-only passwords

Argo CD has no "init user" env vars like Langfuse. Both passwords are bcrypt
hashes passed at install time from your shell, so they never land in Git:

```bash
hash() { htpasswd -nbBC 10 "" "$1" | tr -d ':\n' | sed 's/$2y/$2a/'; }
read -rs -p "admin password: " ADMIN_PWD; echo
read -rs -p "viewer password: " VIEWER_PWD; echo
now=$(date -u +%FT%TZ)
helm upgrade argocd argo/argo-cd -n argocd --version 10.10.1 \
  -f deploy/argocd/install-values.yaml \
  --set-string configs.secret.argocdServerAdminPassword="$(hash "$ADMIN_PWD")" \
  --set-string configs.secret.argocdServerAdminPasswordMtime="$now" \
  --set-string 'configs.secret.extra.accounts\.viewer\.password'="$(hash "$VIEWER_PWD")" \
  --set-string 'configs.secret.extra.accounts\.viewer\.passwordMtime'="$now"
kubectl -n argocd delete secret argocd-initial-admin-secret --ignore-not-found
```

`viewer` maps to `role:readonly`: it sees every Application, diff and log but
cannot sync, delete or edit. Anonymous access and `exec` are off.

## Using it

```bash
kubectl get applications.argoproj.io -n argocd
```

Drift that remains is intentional until reviewed. Do not enable auto-sync on
`llm-runtime` (GPU pod, a restart means a long model reload).
