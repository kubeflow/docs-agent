# docs-agent chart

Helm release for the application layer: the MCP Deployment/Service/ConfigMap,
Kagent ModelConfigs, RemoteMCPServer, and the Agents. It intentionally does not
own the public gateway or the GPU InferenceServices.

Models and agents are plain values. `kagent.models` renders one ModelConfig per
entry. Each entry under `kagent.agents` picks a ModelConfig by name, plus its
prompt file and MCP tools:

| Agent | ModelConfig | Thinking |
|---|---|---|
| `kubeflow-docs-agent` | `flo-llm` | server default |
| `kubeflow-docs-agent-think` | `flo-llm-think` (`reasoningEffort: low`, 8192 tokens) | on, when the model supports it |
| `kubeflow-debug-agent` | `flo-llm` | server default |

The docs agents get `search_kubeflow_docs` and `search_github_issues` only;
`search_kubeflow_code` stays with the debug agent, so code results do not
crowd out docs for conceptual questions.

Both ModelConfigs call the `llm-runtime` release (`llm-stable`, model name
`flo-llm`). Which model answers is chosen there by its `model` value, so a
model switch never changes this chart. The frontend's Think toggle switches
between the two docs agents. Eval results are in `tests/eval/SLM_EVALS.md`.

CI sets the MCP image on every deploy. A manual upgrade without
`mcp.image.repository` and `mcp.image.tag` falls back to the chart default
image, so pass the image that is live (see below).

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
- A normal chart upgrade never owns or restarts the model; that is the separate
  `llm-runtime` release.

Rollback is release-local:

```bash
helm history docs-agent -n docs-agent
helm rollback docs-agent <revision> -n docs-agent --wait --timeout 10m
```
