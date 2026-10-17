# Argo CD

Argo CD deploys everything in this repo. The root Application `flo-apps`
(`root.yaml`) keeps the Applications in `apps/` in line with `main`; each
Application decides whether it syncs on its own.

| Application | Tracks | Sync |
|---|---|---|
| `docs-agent-staging` | `main`, image = each commit's short SHA | automatic; every merge runs the staging eval |
| `docs-agent`, `gateway-guardrails` | the commit in `# prod-revision` | automatic, so merging a promotion PR deploys it |
| `otel-collector`, `langfuse-retention` | `main` | automatic |
| `llm-runtime`, `llm-judge` | `main` | manual: a sync restarts the GPU pod and reloads the model |
| `langfuse` | chart 2.1.0 + `main` values | manual: stateful, review the diff first |

## Promoting to production

Every merge to `main` builds `ghcr.io/kubeflow/mcp-kubeflow-docs:<short sha>` and
`ghcr.io/kubeflow/kubeflow-session-issuer:<short sha>`, and staging deploys and
evaluates that commit. Production moves when someone promotes a commit, usually
every week or two:

```bash
deploy/promote.sh            # promote the tip of main
deploy/promote.sh 9145c31    # or a specific commit
deploy/promote.sh --open     # also push to your fork and open the PR
```

The script checks that both images exist, moves the `# prod-revision` lines and
writes the PR description: the commits since the current production and a table
for the staging eval of that commit (Langfuse project flo-eval, Experiments, run
named `<short sha> …`). Reviewers check the eval and approve; after merge, Argo CD
syncs production to that commit. To roll back, revert the promotion PR.

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
| `langfuse` | `langfuse-app-credentials` (salt, encryption key, NextAuth secret), `langfuse-postgresql-auth`, `langfuse-clickhouse-auth`, `langfuse-redis-auth`, `langfuse-s3-auth` | Langfuse app secrets and its data stores |
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
kubectl apply -f deploy/argocd/root.yaml       # flo-apps applies everything in deploy/argocd/apps/
```

## Admin and read-only passwords

Argo CD has no "init user" env vars like Langfuse. Both passwords are bcrypt
hashes passed at install time from your shell, so they never land in Git:

```bash
hash() { htpasswd -nbBC 10 "" "$1" | tr -d ':\n' | sed 's/$2y/$2a/'; }
read -rs -p "admin password: " ADMIN_PWD; echo
read -rs -p "viewer password: " VIEWER_PWD; echo
now=$(date -u +%FT%TZ)
helm upgrade argocd argo/argo-cd -n argocd --version 10.10.1 --force-conflicts \
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
argocd app diff langfuse --grpc-web         # read-only `viewer` login is enough
```

Do not enable auto-sync on `llm-runtime` or `llm-judge` (GPU pod, a restart
means a long model reload) or on `langfuse`.

## Langfuse data retention

Self-hosted Langfuse keeps data forever unless you pay for its Enterprise
retention, so `deploy/platform/langfuse-retention` runs a weekly CronJob
(Sundays 04:00 UTC). It deletes traces, observations, scores and raw events
older than each project's window, in ClickHouse and in the S3 store:

| Project | Keeps |
|---|---|
| Flo-live (chat traces, user feedback) | 4 weeks |
| flo-eval (staging eval runs) | 2 weeks; each release's scores stay in its promotion PR |

Datasets (the eval questions) are never deleted. Windows are in
`configmap.yaml`; set `DRY_RUN` to `"true"` in the CronJob to only print counts.

## Known gaps

- **In-cluster access control is not enforced.** The Istio AuthorizationPolicies
  for `ml-infra` (LLM, judge, Milvus, embeddings) only take effect on pods with
  an Istio sidecar, and `docs-agent` and `ml-infra` have no sidecar injection.
  Verified 2026-10-14: a pod in `default` reaches `llm-stable`, `llm-judge` and
  Milvus. The public gateway (route allow-list, session tokens, rate limits) is
  enforced and unaffected. The cluster CNI (`vcn-native-ip-cni`, no Calico) is
  not expected to enforce NetworkPolicy either. Planned fix: enable sidecar
  injection namespace by namespace, `ml-infra` first, testing chat after each.
