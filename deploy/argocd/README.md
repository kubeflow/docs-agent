# Argo CD (observe-only)

Argo CD watches the Helm releases (`docs-agent`, `gateway-guardrails`,
`llm-runtime`, `llm-judge`) against `kubeflow/docs-agent` `main`. None of the
Applications has a `syncPolicy`: Argo reports drift, and a maintainer clicks
Sync to deploy. GPU releases (`llm-runtime`, `llm-judge`) stay manual.

## What lives where

| Kind of data | Where | Who can change it |
|---|---|---|
| Charts, Argo Applications | `docs-agent-mcp/charts/`, `deploy/argocd/` | PR + review |
| Per-environment values (domain, CORS origins, image overrides, ingress hosts) | `deploy/envs/<env>/` in this repo, read by Argo as a second source (`$values/deploy/envs/prod/<app>.yaml`) | PR + review |
| Secrets (API keys, Langfuse keys, passwords, pull secrets) | Kubernetes Secrets only, referenced by name; inventory and local backup in `deploy/secrets/` | cluster admins |

Values files are public, so nothing personal or secret goes in them. The ACME
contact email, for example, lives only in the cluster's ClusterIssuer
(`gateway.tls.acme.createIssuer: false`). Argo CD cannot read Helm values from
a Kubernetes Secret on its own, so secret values are never Helm values: the
chart takes a Secret name.

## Install / upgrade

```bash
helm repo add argo https://argoproj.github.io/argo-helm
helm upgrade --install argocd argo/argo-cd -n argocd --create-namespace --version 10.10.1 \
  -f deploy/argocd/install-values.yaml
kubectl apply -f deploy/envs/prod/ingress/        # public HTTPS for Langfuse and Argo (needs A records)
kubectl apply -f deploy/platform/llm-judge/authz.yaml
kubectl apply -f deploy/argocd/apps/
```

## Admin password

Argo CD has no "init user" env vars like Langfuse. Two options:

1. Default: the chart generates a random one in `argocd-initial-admin-secret`.
   Read it, log in, change it, then delete that secret.
2. Set it at install time. Pass a bcrypt hash from your shell so it never lands in Git:

```bash
read -rs ARGO_PWD
helm upgrade argocd argo/argo-cd -n argocd --version 10.10.1 \
  -f deploy/argocd/install-values.yaml \
  --set-string configs.secret.argocdServerAdminPassword="$(htpasswd -nbBC 10 "" "$ARGO_PWD" | tr -d ':\n' | sed 's/$2y/$2a/')" \
  --set-string configs.secret.argocdServerAdminPasswordMtime="$(date -u +%FT%TZ)"
kubectl -n argocd delete secret argocd-initial-admin-secret --ignore-not-found
```

Read-only access for mentors: add a local account in `configs.cm` (`accounts.viewer: login`)
and `configs.rbac` (`g, viewer, role:readonly`), then set its password with
`argocd account update-password`.

## Using it

```bash
kubectl get applications.argoproj.io -n argocd
```

Drift that remains is intentional until reviewed. Do not enable auto-sync on
`llm-runtime` (GPU pod, a restart means a long model reload).
