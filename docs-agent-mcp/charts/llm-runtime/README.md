# llm-runtime chart

Separate Helm release for the GPU-serving plane. Keeping it outside the
`docs-agent` release guarantees that an MCP/prompt/widget merge does not touch
the InferenceService.

One release serves one model on one GPU: a KServe InferenceService (Standard
mode, plain vLLM container), a weight-cache PVC, and the stable Service
`llm-stable`. Clients call `http://llm-stable.ml-infra.svc.cluster.local/v1`
with the model name `servedModelName` (`flo-llm`), whichever model is active.

## Choosing the model

`values.yaml` holds a `catalog` of validated profiles; `model` picks one. A
profile sets the image, model ID and revision, quantization, context length,
GPU memory share, tool and reasoning parsers, thinking default, extra vLLM
flags, CPU, RAM, and `/dev/shm`. Placement, cache, and Service are shared.

Switching is a one-value change, reviewed in a PR like any other:

```yaml
model: <catalog key>
```

```bash
helm upgrade --install llm-runtime ./docs-agent-mcp/charts/llm-runtime \
  --namespace ml-infra --atomic --cleanup-on-fail --timeout 10m --history-max 5
kubectl rollout status deployment/llm-predictor -n ml-infra --timeout=1800s
helm test llm-runtime -n ml-infra
```

CI runs the same commands when the workflow is dispatched with
`deploy_llm=true`. An ordinary merge never restarts the model.

To add a profile: add an entry under `catalog`, evaluate it with
`tests/eval/slm_tool_eval.py`, record the result in `tests/eval/SLM_EVALS.md`,
then switch `model`. Credentials never go in values: a gated model reads its
token from the Secret named in `hfToken.existingSecret`.

## Operations

`deploymentStrategy: Recreate` is intentional: each model holds a whole GPU, so
a rolling update cannot schedule old and new replicas simultaneously. Treat a
model or image change as a maintenance window with a successful `helm test`.

The cache PVC is annotated `helm.sh/resource-policy: keep` and is shared by all
profiles, so uninstalling or switching back does not re-download weights.

Rollback is either `helm rollback llm-runtime <revision> -n ml-infra` or
setting `model` back and upgrading. Clients need no change in either case.

The Helm test pod runs in `docs-agent` and calls the stable service in
`ml-infra`. This checks the same namespace-to-LLM authorization path used by
Kagent (the `gateway-guardrails` mesh policy), rather than testing only from
inside the serving namespace.

Version the chart with SemVer. Bump PATCH for safe values/template fixes, MINOR
for backward-compatible runtime features or new catalog profiles, and MAJOR
for storage contract or deployment-mode changes.
