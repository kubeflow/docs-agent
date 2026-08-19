# docs-agent chart

Helm release for the application layer: the MCP Deployment/Service/ConfigMap,
Kagent ModelConfig, RemoteMCPServer, and the Docs/Debug Agents. It intentionally
does not own the public gateway or the GPU InferenceService.

## Install or upgrade

```bash
helm upgrade --install docs-agent ./docs-agent-mcp/charts/docs-agent \
  --namespace docs-agent --create-namespace \
  --atomic --cleanup-on-fail --wait --timeout 10m --history-max 10 \
  --set-string mcp.image.tag=<immutable-image-tag>
helm test docs-agent -n docs-agent
```

For the first migration from `kubectl apply`, server-dry-run the render and use a
Helm version that supports `--take-ownership`. Do not combine the first adoption
with `--atomic`: a failed atomic *install* can uninstall resources it just
adopted. Established releases use `--atomic --cleanup-on-fail`. The chart
references the existing `mcp-server-secret`; it never stores the Milvus password.

## Versioning

- `Chart.yaml version` follows SemVer for template/default-value changes.
- `appVersion` follows the public docs-agent application release.
- Production values pin the MCP image to an immutable SemVer tag or digest.
- A normal chart upgrade never owns or restarts Qwen; that is the separate
  `qwen-runtime` release.

Rollback is release-local:

```bash
helm history docs-agent -n docs-agent
helm rollback docs-agent <revision> -n docs-agent --wait --timeout 10m
```
