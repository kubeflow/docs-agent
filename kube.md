# kubectl quick reference — docs-agent on shared Kubeflow cluster

Your current context (`kubectl config current-context`) should point at the shared OCI Kubeflow cluster. Re-check anytime:

```bash
kubectl config current-context
kubectl cluster-info
```

---

## Namespace boundaries (keep Kubeflow safe)

| Area | Typical namespaces | What to touch for docs-agent iteration |
|------|-------------------|----------------------------------------|
| Docs-agent stack | `docs-agent` | Deployments, Services, Helm releases, ConfigMaps tied to your app/Milvus/kagent |
| Kubeflow control plane | `kubeflow`, `kubeflow-system`, `istio-system`, `knative-serving`, … | Prefer **do not delete** wholesale — shared platform |
| User sandboxes | `user` | Other contributors’ workspaces — avoid |

**Important:** Prefer **never** draining or deleting **cluster nodes** when you only want to reset docs-agent workloads. Nodes are shared; removing a node hurts everything on it (including Kubeflow).

---

## Core inspection — `docs-agent` namespace

```bash
# Everything the shortcut "all" knows about
kubectl get all -n docs-agent

# Pods with node placement (helps see which worker runs your pods)
kubectl get pods -n docs-agent -o wide

# Detailed pod state / scheduling problems
kubectl describe pod -n docs-agent <pod-name>

# Logs (deployment convenience)
kubectl logs -n docs-agent deploy/kubeflow-docs-agent --tail=100 -f
kubectl logs -n docs-agent deploy/mcp-kubeflow-docs --tail=100 -f
```

---

## Labels, owners, and “what Helm made”

```bash
# Labels on all workloads
kubectl get deploy,sts,svc -n docs-agent --show-labels

# Helm-managed Deployments carry common labels — same for Milvus chart
kubectl get deploy -n docs-agent -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.labels.app\.kubernetes\.io/managed-by}{"\n"}{end}'
```

See installed Helm releases (names overlap with chart secrets `sh.helm.release.v1.*`):

```bash
helm list -n docs-agent
helm status kagent -n docs-agent
helm status my-release -n docs-agent
```

---

## Services and endpoints

```bash
kubectl get svc -n docs-agent -o wide
kubectl get endpoints -n docs-agent

# Resolve DNS-style service names inside the cluster
# <svc>.docs-agent.svc.cluster.local
```

---

## ConfigMaps, Secrets (no secret values dumped here)

```bash
kubectl get configmap -n docs-agent
kubectl get secret -n docs-agent

kubectl describe configmap mcp-server-config -n docs-agent
kubectl describe configmap etl-pipeline-config -n docs-agent

# Inspect keys only (values stay base64 — still sensitive)
kubectl get secret mcp-server-secret -n docs-agent -o jsonpath='{.data}' | head -c 200; echo
```

---

## Storage

```bash
kubectl get pvc -n docs-agent
kubectl get pv
```

(If `pvc` is empty, Milvus/Minio may be using `emptyDir` or cluster-specific storage classes — check the StatefulSet/Deployment specs.)

---

## KServe / inference (if you add `InferenceService` later)

```bash
kubectl get inferenceservices.serving.kserve.io -n docs-agent
kubectl api-resources --api-group=serving.kserve.io
```

---

## Kubeflow Pipelines — Argo Workflows

Pipelines compile to Argo `Workflow` objects (namespace-scoped):

```bash
# Recent workflow runs in a namespace
kubectl get workflows.argoproj.io -n kubeflow
kubectl get workflows.argoproj.io -n user
kubectl get workflows.argoproj.io -n docs-agent

# Drill into one run
kubectl describe workflow <wf-name> -n <namespace>

# Scheduled workflows (classic KFP)
kubectl get scheduledworkflows -n kubeflow
kubectl get scheduledworkflows -n user
```

---

## Platform health (read-only snapshots)

```bash
kubectl get pods -n kubeflow
kubectl get pods -n kubeflow-system
kubectl get pods -n istio-system | head
kubectl top pods -n docs-agent   # needs metrics-server
```

---

## Discovery helpers

```bash
# List every namespaced resource type you can query
kubectl api-resources --namespaced=true

# Explain a field (e.g. Deployment)
kubectl explain deployment.spec.template.spec.containers
```

---

## MCP image (GitHub Actions) and Milvus / pipelines

### 1) Build and push MCP image (no local Docker)

Workflow: `.github/workflows/build-mcp-server.yml` — builds from `kagent-feast-mcp/mcp-server/Dockerfile`, pushes to **`ghcr.io/<github-username-or-org>/mcp-kubeflow-docs`**.

**Prerequisites:** Repository **Actions** enabled; for private images, package read access for the cluster pull secret if needed.

**Run from GitHub:** Actions → **Build MCP server (GHCR)** → **Run workflow**. Optionally set **image_tag** (for example `all-tools-v3`). Every run also pushes a tag equal to the short **git SHA**.

**After push:** Point the deployment at the new tag (example — replace namespace and tag):

```bash
kubectl set image deployment/mcp-kubeflow-docs \
  mcp-server=ghcr.io/<owner>/mcp-kubeflow-docs:<sha-or-tag> \
  -n docs-agent
kubectl rollout status deployment/mcp-kubeflow-docs -n docs-agent
```

**MCP ↔ Milvus alignment:** The server defaults to collections **`docs_rag`**, **`issues_rag`**, **`code_rag`**. Pipeline parameters use the same defaults. Set `MILVUS_URI` / `MILVUS_USER` / `MILVUS_PASSWORD` / `COLLECTION_NAME` / `ISSUES_COLLECTION_NAME` / `CODE_COLLECTION_NAME` on the deployment if your cluster differs (see `kagent-feast-mcp/mcp-server/server.py`).

---

### 2) Clear current docs-agent Milvus collections (full re-index)

Identify running Milvus endpoint (Helm release is often `my-release-milvus` in `docs-agent`):

```bash
kubectl get svc -n docs-agent | grep -i milvus
```

Port-forward (from your laptop):

```bash
kubectl port-forward svc/my-release-milvus -n docs-agent 19530:19530
```

Drop the RAG collections used by the MCP (names must match your pipelines and env vars).

**If Docker Desktop is not running** (Mac often shows `docker.sock: connect: no such file or directory`), skip `docker run` and use **A** or **B** below.

**A — Port-forward + local Python**

```bash
# Terminal 1
kubectl port-forward svc/my-release-milvus -n docs-agent 19530:19530
```

```bash
# Terminal 2
pip install pymilvus
python -c "
from pymilvus import connections, utility
connections.connect('default', host='127.0.0.1', port='19530', user='root', password='Milvus')
for c in ['docs_rag', 'issues_rag', 'code_rag']:
    if utility.has_collection(c):
        utility.drop_collection(c)
        print('Dropped', c)
    else:
        print('No collection', c)
"
```

**B — Ephemeral pod in `docs-agent` (no local Docker)**

```bash
kubectl run milvus-drop-rag --rm -it --restart=Never -n docs-agent \
  --image=python:3.11-slim \
  -- bash -c 'pip install -q pymilvus && python -c "
from pymilvus import connections, utility
connections.connect(\"default\", host=\"my-release-milvus.docs-agent.svc.cluster.local\", port=\"19530\", user=\"root\", password=\"Milvus\")
for c in [\"docs_rag\", \"issues_rag\", \"code_rag\"]:
    if utility.has_collection(c):
        utility.drop_collection(c)
        print(\"Dropped\", c)
    else:
        print(\"No collection\", c)
"'
```

**Optional — Docker on laptop** (daemon must be running; Linux `--network host` only):

```bash
docker run --rm --network host python:3.11-slim bash -c \
  "pip install -q pymilvus && python -c \"
from pymilvus import connections, utility
connections.connect('default', host='127.0.0.1', port='19530', user='root', password='Milvus')
for c in ['docs_rag', 'issues_rag', 'code_rag']:
    if utility.has_collection(c):
        utility.drop_collection(c)
        print('Dropped', c)
    else:
        print('No collection', c)
\""
```

Adjust **host** if you forward a different local port; adjust **user/password** if your Milvus secret differs.

---

### 3) Compile pipeline packages and run on Kubeflow

From a dev environment with `kfp` and project deps (Kubeflow Notebook, CI job, or lightweight cloud VM — not necessarily your Mac):

```bash
pip install kfp ...
cd pipelines
python kubeflow-pipeline.py      # writes compiled YAML for docs RAG
python issues-pipeline.py         # github_issues_rag_pipeline.yaml
python code-pipeline.py           # code_rag_pipeline.yaml
```

Upload the generated YAML in the **Kubeflow Pipelines** UI → **Upload pipeline**, then **Create run**.

**Find your cluster’s public URL (example discovery):**

```bash
kubectl get svc -n istio-system istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}{"\n"}'
```

Use **`http://<EXTERNAL-IP>/`** for the Central Dashboard (may redirect to auth). Pipelines UI is usually **`http://<EXTERNAL-IP>/pipeline/`** (exact path can vary slightly by Kubeflow version). If TLS or OAuth is configured, use **`https`** and the hostname your admins documented instead of raw IP.

**Defaults for “index more”:**

| Pipeline | Parameters to scale up |
|----------|-------------------------|
| Issues (`github_issues_rag_pipeline`) | `repos` — add comma-separated org/repos; `max_issues_per_repo`; optional `labels` / `state` |
| Code (`code_rag_pipeline`) | `repos` — e.g. add `kubeflow/kubeflow`; `directory_paths` — broader paths under those repos (CSV); `file_extensions` — includes `yaml,yml,py,json` |

**Milvus hostname in runs:** Pipeline defaults target **`my-release-milvus.docs-agent.svc.cluster.local`** (Helm Milvus in `docs-agent`). Override the **milvus_host** run parameter only if your service DNS name differs.

**Validate the ingestion / Milvus idempotency fix:** Run the same pipeline twice with overlapping data — second run should **not** wipe unrelated chunks; logs in the `store_*_milvus` step should mention deletes by `file_unique_id` batches and inserts without dropping the whole collection. For logs:

```bash
kubectl get workflows.argoproj.io -n <kfp-namespace> --sort-by=.metadata.creationTimestamp | tail -5
kubectl logs -n <kfp-namespace> <workflow-step-pod> -c main --tail=400
```

---

## Troubleshooting: pipeline pods blocked (`PodDefaults` admission webhook EOF)

If workflows fail immediately with **`admission-webhook-deployment.kubeflow.org`** / **`Post … admission-webhook-service … EOF`**, the webhook usually has **no healthy pods**. Namespace **`kubeflow`** may enforce **`pod-security.kubernetes.io/enforce=restricted`**, while the stock webhook Deployment ships **without** a restrictive `securityContext`, so **pods never schedule**.

Check:

```bash
kubectl get deploy admission-webhook-deployment -n kubeflow
kubectl get pods -n kubeflow -l app=poddefaults
kubectl describe rs -n kubeflow -l app=poddefaults | tail -20
```

Fix (patch webhook Deployment so PSA restricted accepts it — **`runAsUser`** may need tuning if the image fails startup):

```bash
kubectl patch deployment admission-webhook-deployment -n kubeflow --type=json -p='[
  {"op": "replace", "path": "/spec/template/spec/securityContext", "value": {"runAsNonRoot": true, "runAsUser": 1000, "seccompProfile": {"type": "RuntimeDefault"}}},
  {"op": "add", "path": "/spec/template/spec/containers/0/securityContext", "value": {"allowPrivilegeEscalation": false, "capabilities": {"drop": ["ALL"]}, "readOnlyRootFilesystem": false}}
]'
kubectl rollout status deployment/admission-webhook-deployment -n kubeflow
```

Persist the same fields in your **Terraform / manifests** so upgrades do not revert this.

---

## Troubleshooting: `ImageInspectError` / “short name mode is enforcing … ambiguous list”

Some nodes run **containerd** with **short name mode enforcing**. Unqualified images like `python:3.9` or `pytorch/pytorch:…` can fail inspection (`Failed to inspect image ""` or **ambiguous list**). Use fully qualified refs in **`@dsl.component(base_image=...)`**, e.g. **`docker.io/library/python:3.9`** and **`docker.io/pytorch/pytorch:…`** (same pattern as `code-pipeline.py`).

---

## Future: resetting **only** docs-agent workloads (reference — do not run blindly)

Discuss with mentors before teardown. Order of operations usually matters:

1. Understand what is Helm-owned vs standalone manifests (`helm list -n docs-agent`).
2. `mcp-kubeflow-docs` and some agent objects may **not** be in the Helm chart — track them via `kubectl get all -n docs-agent -o yaml` backups.
3. Uninstall Helm releases (`kagent`, `my-release`, …) only after confirming dependencies; `kagent-crds` affects cluster-wide CRDs.
4. Deleting namespace `docs-agent` removes Services and workload identity for that DNS zone — Kubeflow core namespaces remain untouched **if you only delete `docs-agent`**.

Until you need a lab reset, **image tag updates** (`kubectl set image …` / `helm upgrade … --set image.tag=…`) are usually enough to test new pipelines and server code against the existing Kubeflow install.