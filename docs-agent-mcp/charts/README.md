# Helm release strategy

The stack is split into three releases so each upgrade has a small, explicit
blast radius:

| Release | Owns | Automatic on main | Downtime expectation |
|---|---|---:|---|
| `docs-agent` | MCP Deployment/Service/ConfigMap and Kagent resources | Yes | None; MCP uses maxUnavailable=0 |
| `gateway-guardrails` | Istio routing, TLS, auth, rate limits, mesh policy | No | Normally none; promote separately |
| `qwen-runtime` | PVC, ServingRuntime, InferenceService, stable Service | **No** | Expected; single GPU uses Recreate |

KFP ingestion remains a separate operator-approved data operation. No Helm
install or upgrade submits a pipeline run.

## Promotion flow

1. Pull request: Python tests, Helm lint, and deterministic `helm template`.
2. Merge: build an immutable SHA-tagged MCP image and upgrade only `docs-agent`
   (atomic after the one-time ownership transfer).
3. Production SemVer promotion: dispatch the workflow with
   `release_version=X.Y.Z`; the image is tagged with that version and installed
   through the same chart.
4. GPU runtime: dispatch separately with `deploy_kserve=true` during an approved
   downtime window. An ordinary merge cannot enter this path.

Use SemVer independently for each chart:

- PATCH: backward-compatible fixes or safe default tuning.
- MINOR: new optional resources/features or backward-compatible value keys.
- MAJOR: renamed/removed values, ownership changes, model/storage migrations,
  or other operator action.

`appVersion` records the application/model release; deployment values still pin
immutable image tags. Never put credentials in a values file—charts reference
existing Kubernetes Secrets.

## First adoption from kubectl

The existing resources are not Helm-owned. The first upgrade uses Helm 3.18+
with `--take-ownership`; subsequent upgrades no longer need that flag. Helm
3.18 or newer is required because it fixes ownership adoption for custom
resources such as Kagent Agents and KServe InferenceServices. Render and
server-dry-run before adoption:

```bash
helm lint docs-agent-mcp/charts/docs-agent
helm template docs-agent docs-agent-mcp/charts/docs-agent -n docs-agent \
  --set-string mcp.image.tag=<tag> \
  | kubectl apply -n docs-agent --dry-run=server -f -
```

The first ownership transfer is deliberately non-atomic after server dry-run;
this prevents a failed atomic install from deleting adopted resources:

```bash
helm upgrade --install docs-agent docs-agent-mcp/charts/docs-agent \
  -n docs-agent --create-namespace --take-ownership \
  --wait --timeout 10m --history-max 10 \
  --set-string mcp.image.repository=<repo> \
  --set-string mcp.image.tag=<immutable-tag>
```

After revision 1 exists, CD uses atomic upgrades and retains ten revisions:

```bash
helm upgrade docs-agent docs-agent-mcp/charts/docs-agent \
  -n docs-agent --atomic --cleanup-on-fail --wait --timeout 10m --history-max 10 \
  --set-string mcp.image.repository=<repo> \
  --set-string mcp.image.tag=<immutable-tag>
helm test docs-agent -n docs-agent
```

## Rollback

Application rollback never touches Qwen:

```bash
helm history docs-agent -n docs-agent
helm rollback docs-agent <revision> -n docs-agent --wait --timeout 10m
```

Runtime rollback is deliberately separate and should be done in a downtime
window. The Hugging Face cache PVC has `helm.sh/resource-policy: keep`, so a
release rollback or uninstall does not erase model weights.
