# Argo CD

Argo CD deploys everything in this repo. The root Application `flo-apps`
(`root.yaml`) keeps the Applications in `apps/` in line with `main`; each
Application decides whether it syncs on its own.

| Application | Tracks | Sync |
|---|---|---|
| `docs-agent-staging` | `main`, image = each commit's short SHA | automatic; every merge runs the staging eval |
| `docs-agent-pr-<n>`, `docs-agent-pr-<n>-eval` | a PR labelled `eval` (ApplicationSets `docs-agent-preview`, `docs-agent-pr-eval`) | automatic; exist only while the PR's eval runs |
| `preview-namespace` | `main` | automatic; the `docs-agent-preview` Namespace and its Pod Security level |
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

**First promotion.** MCP images built from #266 up to the commit that fixed the
Dockerfile (`flo_tracing.py` was not copied) crash on start, so the pin in this
directory must not be synced. After this change merges, promote its merge commit
(`deploy/promote.sh <merge sha> --open`), merge that PR, and only then apply
`root.yaml`. Until then production keeps running `0e49ebd`.

## Pre-merge eval (`/eval`)

The staging eval runs after a merge. To see whether a PR helps before merging it, a
maintainer comments `/eval` on the PR. It is not a CI check: it uses the shared GPU
model, so it only runs when an OWNERS approver (or a repo admin/maintainer) asks.

1. `.github/workflows/pr-eval.yaml` builds the PR's MCP image as
   `ghcr.io/kubeflow/mcp-kubeflow-docs:pr-<n>-<sha8>` and labels the PR `eval` and
   `eval-sha:<sha>`. The second label pins the commit the maintainer reviewed:
   commits pushed afterwards are not deployed. One PR holds the preview at a time; a
   second `/eval` is told to wait, and an eval holding it for over 3 hours is cleared.
2. The `docs-agent-preview` ApplicationSet deploys the pinned commit (its chart,
   prompts and image, with the values in the ApplicationSet itself, from `main`) into
   `docs-agent-preview`, under the AppProject `docs-agent-preview`: that namespace
   only, the chart's own kinds only, nothing cluster-scoped. The namespace enforces
   Pod Security `baseline` (no privileged pods, host paths or host networking).
3. The `docs-agent-pr-eval` ApplicationSet runs `docs-agent-mcp/charts/pr-eval` from
   `main` in `docs-agent-staging`. Its Job (`tests/eval/pr_eval.py`) waits until the
   preview runs the PR's image, runs the eval against the preview and then against
   staging (`main`), and comments the comparison on the PR: the metrics side by side,
   the cases that got worse or better, the judge's reasons and a public Langfuse trace
   per case, so contributors can see where an answer went wrong without an account.
4. The Job removes both labels, and Argo CD deletes both Applications and the preview.

Posting needs the optional Secret `pr-eval-github` (below); create or rotate it
without any code change:

```bash
read -s T; kubectl -n docs-agent-staging create secret generic pr-eval-github \
  --from-literal=token="$T" --dry-run=client -o yaml | kubectl apply -f -; unset T
```

Without it, or if GitHub rejects the token, the report is only in the Job log
(`kubectl -n docs-agent-staging logs job/pr-eval-<n>-<sha8>`) and the labels stay
until removed by hand or by the next `/eval` after 3 hours.

The grading (script, dataset, judge) always comes from `main`, and main is measured
first, so a PR cannot change how it is scored or load the shared services during
main's run. Changes that need a re-index (ingestion, chunking) are not covered: the
preview reads the production collections.

The PR's code does run in the cluster, so review the diff before commenting `/eval`,
as for `/ok-to-test` elsewhere. The AppProject and Pod Security above keep it in its
namespace and off the nodes, but not away from shared services: this cluster's CNI
(OCI VCN-native) does not enforce NetworkPolicy, and Milvus runs without
authorization (`common.security.authorizationEnabled` is off), so any pod, a preview
included, could write to or drop the production collections. Enabling Milvus
authorization with a read-only user for the MCP server is the fix and a separate
change; until then, the review before `/eval` is the control. The `eval` labels are
the trigger, so anyone with triage access to the repository could add them by hand;
keep that group to maintainers.

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
| `docs-agent-preview` | `mcp-server-secret` (`MILVUS_PASSWORD`), `mcp-langfuse-keys` (evaluation project) | the PR under `/eval`; PR code can read these. Copies of staging's while Milvus authorization is off; a read-only Milvus user once it is on |
| `docs-agent-staging` | `pr-eval-github` (key `token`), optional | GitHub token that may comment on and label PRs in kubeflow/docs-agent (fine-grained: Issues + Pull requests read/write; classic: `public_repo`). Lets the pr-eval Job post its report and remove the labels. Without it the report is only in the Job log and the eval still succeeds. |
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
