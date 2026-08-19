# qwen-runtime chart

Separate Helm release for the GPU-serving plane. Keeping it outside the
`docs-agent` release guarantees that an MCP/prompt/widget merge does not touch
the InferenceService.

## First migration

The legacy Knative service holds the only GPU. Stop it before the first Helm
adoption, then adopt the existing Standard-mode resources:

```bash
kubectl annotate inferenceservice/qwen-llm -n ml-infra \
  serving.kserve.io/stop=true --overwrite
kubectl wait --for=condition=Stopped=True inferenceservice/qwen-llm \
  -n ml-infra --timeout=600s

helm upgrade --install qwen-runtime ./docs-agent-mcp/charts/qwen-runtime \
  --namespace ml-infra --create-namespace --take-ownership \
  --timeout 10m --history-max 5
kubectl rollout status deployment/qwen-llm-standard-predictor \
  -n ml-infra --timeout=1800s
helm test qwen-runtime -n ml-infra
```

`deploymentStrategy: Recreate` is intentional: the cluster has one GPU, so a
rolling update cannot schedule old and new replicas simultaneously. Only an
explicit runtime release or a change under this chart should run the upgrade.
The Hugging Face PVC is annotated `helm.sh/resource-policy: keep`, so uninstall
does not delete the model cache.

The Helm test pod runs in `docs-agent` and calls the stable service in
`ml-infra`. This checks the same namespace-to-LLM authorization path used by
Kagent, rather than testing only from inside the serving namespace.

Version the chart with SemVer. Bump PATCH for safe values/template fixes, MINOR
for backward-compatible runtime features, and MAJOR for migrations such as a
model ID, storage contract, or deployment-mode change.

Do not use `--atomic` on the first ownership-transfer install: if adoption
fails, atomic uninstall could remove resources that existed before Helm. After
revision 1 exists, use `--atomic --cleanup-on-fail` for upgrades.
