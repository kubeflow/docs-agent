# Argo CD (observe-only)

Argo CD watches the Helm releases (`docs-agent`, `gateway-guardrails`,
`llm-runtime`, `llm-judge`) against `kubeflow/docs-agent` `main`. None of the
Applications has a `syncPolicy`: Argo reports drift, and a maintainer clicks
Sync to deploy. GPU releases (`llm-runtime`, `llm-judge`) stay manual.

## What lives where

| Kind of data | Where | Who can change it |
|---|---|---|
| Charts, Argo Applications, generic values | this repo | PR + review |
| Deployment values (domain, ACME contact email, CORS origins, image overrides, ingress hosts) | a private config repo, read by Argo as a second source (`$values/envs/prod/<app>.yaml`) | config-repo admins, by PR |
| Secrets (API keys, Langfuse keys, passwords, pull secrets) | Kubernetes Secrets only, referenced by name from the charts | cluster admins (RBAC on `secrets`); later External Secrets from OCI Vault |

Argo reads the private repo with a read-only deploy key, registered as a
repository Secret (never committed):

```bash
kubectl -n argocd create secret generic repo-docs-agent-deploy \
  --from-literal=type=git --from-literal=url=git@github.com:<you>/docs-agent-deploy.git \
  --from-file=sshPrivateKey=$HOME/.ssh/docs_agent_deploy_ro
kubectl -n argocd label secret repo-docs-agent-deploy argocd.argoproj.io/secret-type=repository
```

(With an SSH key, use the `git@github.com:` URL in the Applications as well.)

Argo CD cannot read Helm values from a Kubernetes Secret on its own. Values
that are truly secret stay out of Helm values: the chart takes a Secret name
(`existingSecret`), and the Secret is created by an admin or by External
Secrets.

## Install / upgrade

```bash
helm repo add argo https://argoproj.github.io/argo-helm
helm upgrade --install argocd argo/argo-cd -n argocd --create-namespace --version 10.10.1 \
  -f deploy/argocd/install-values.yaml --set global.domain=<your Argo host>
# public HTTPS: apply the Gateway/VirtualService from your config repo (needs an A record for the host)
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
