# LLM runtime: decisions

Status: **live since 2026-10-04 (rollout below).**
Scope: [PR #261](https://github.com/kubeflow/docs-agent/pull/261). Last updated 2026-10-04.

## Summary

| # | Topic | Decision |
|---|---|---|
| 1 | LLM chart | One model-neutral chart, `llm-runtime` (renamed from `qwen-runtime`). The model is picked by one value, `model:`, from a catalog of profiles in `values.yaml` |
| 2 | Active models | One active model for every agent. Today `gemma`; `qwen` stays in the catalog as a one-value rollback |
| 3 | Engine | Every profile runs on the same pinned vLLM image (v0.30.0) as a KServe custom container. There is no ServingRuntime |
| 4 | Client contract | The model is always served as `flo-llm` at `http://llm-stable.ml-infra.svc.cluster.local/v1`. ModelConfigs, agents and the frontend never change when the model does |
| 5 | Secrets | Credentials only (`hfToken.existingSecret`, `kagent.apiKeySecret`). The model choice is not a secret |
| 6 | Chart versions | `docs-agent` 0.3.0, `llm-runtime` 0.2.0, `gateway-guardrails` 0.2.0 |
| 7 | Mesh access | `gateway-guardrails` takes a list of InferenceService names, so old and new models can both be allowed during a cutover |
| 8 | CI | One manual switch, `deploy_llm`. Lint and render every catalog profile. No model names in the workflow |
| 9 | Tool-choice proxy | Removed; all agents use `tool_choice: auto`. Kept for a later iteration |
| 10 | Langfuse | Removed from this PR; it moves to the observability PR |
| 11 | Evals | One markdown record (`tests/eval/SLM_EVALS.md`); result JSONs and `slm_chat.py` are not committed |
| 12 | Thinking | UI "Think deeper" toggle that switches to a thinking agent (`reasoningEffort: low`) |
| 13 | Docs prompt | Rewritten for Gemma (identity, scope, query rules); A/B-tested with `prompt_lab.py` |

---

## 1. How teams switch LLMs (case study)

- **KubeAI `kubeai/models` chart:** `values.yaml` holds a catalog of pre-validated model profiles, and operators enable one with a single flag (`catalog.<name>.enabled: true`). Its Weaviate tutorial serves Gemma under the name `gpt-3.5-turbo`, so clients never change. ([install models](https://www.kubeai.org/how-to/install-models/), [values.yaml](https://github.com/substratusai/kubeai/blob/main/charts/models/values.yaml), [Weaviate tutorial](https://www.kubeai.org/tutorials/weaviate/))
- **vLLM production-stack chart:** `servingEngineSpec.modelSpec` lists one entry per model. The HF token is a secret reference, not model configuration. ([helm docs](https://docs.vllm.ai/projects/production-stack/en/latest/deployment/helm.html))
- **LiteLLM:** clients call a stable alias (`model_name`), and the backing model is swapped behind it in config. ([routing](https://docs.litellm.ai/docs/routing))
- **Common practice:** the model choice is a values key in Git. That way it is reviewed in a PR, shows in `helm diff`, and is undone with `helm rollback`. Secrets hold only credentials. A Secret is the wrong place for the switch for three reasons: the choice is not sensitive, a Secret hides the change from review, and Helm cannot branch templates on Secret contents at render time.

This chart follows the same pattern: a catalog (KubeAI), one switch, and a stable alias (LiteLLM).

## 2. The `llm-runtime` chart

```text
values: model = gemma ──> catalog.gemma ──> InferenceService llm (vLLM, served as flo-llm)
                                             └─> Service llm-stable :80 /v1
                                                  └─> ModelConfigs flo-llm, flo-llm-think
                                                       └─> docs, docs-think, debug agents
```

**Values** (`docs-agent-mcp/charts/llm-runtime/values.yaml`):

| Key | Meaning |
|---|---|
| `model` | The one switch: a key under `catalog` |
| `servedModelName` | Stable alias clients call (`flo-llm`) |
| `catalog.<name>` | One profile per model: image, HF id and pinned revision, quantization, context length, GPU memory share, tool and reasoning parsers, thinking default, extra args, CPU, RAM and `/dev/shm` |
| `placement` | GPU node selector and tolerations, shared by all profiles. |
| `cache` | One RWO weights PVC (`llm-hf-cache`, 50Gi), kept on uninstall. It holds the weights of every profile that has run |
| `hfToken.existingSecret` | Optional; only for gated models |
| `inferenceService`, `service` | Names, rollout strategy (`Recreate`: one GPU), probes, and stable Service port |

**Templates** contain no model names:
- `inferenceservice.yaml` builds one vLLM container from the selected profile.
- `_helpers.tpl` fails the render if `model` is not a catalog key.
- `pvc.yaml` and `service.yaml` create the weights PVC and the stable Service.
- `tests/model-endpoint.yaml` is the Helm test: a curl pod in `docs-agent`, on kagent's network path, checks that `/v1/models` lists `flo-llm`. CI runs it after every model deploy.
- `NOTES.txt` prints the active profile and endpoint.

**Adding a model** means adding a profile to `catalog`, adding a harness candidate in `tests/eval/slm_tool_eval.py`, running the evals, and recording them in `SLM_EVALS.md`. Switching to it is then a one-line `model:` change plus `deploy_llm`.

**Options considered**

| Option | Verdict |
|---|---|
| A separate chart per model (`slm-runtime`, `values-gemma.yaml`) | Rejected: duplicate charts and model-specific CI switches |
| The same chart installed twice (one release per model) | Rejected: two GPUs held permanently, and agents bound to model-specific endpoints |
| **One release, catalog plus one switch, stable alias** | **Chosen** |
| The model choice in a Secret | Rejected (see the case study) |

## 3. One active model, one engine

- **Single active model.** Gemma think-off matches Qwen on the debug cases (51/60 each) and beats it overall (258 vs 168/291), so the debug agent moves to the same model as Flo. This frees the second A10.
- **Unified engine.** Qwen used KServe huggingfaceserver v0.15.2 (vLLM 0.8.5) through the `llm-runtime` ServingRuntime. No huggingfaceserver release can serve Gemma 4 tool calls, so both profiles now run on `vllm/vllm-openai` v0.30.0 (pinned digest) as a custom container. This is the same pattern `ml-infra/embeddings-service` already uses.
- **Qwen is an unvalidated rollback until it is re-evaluated.** The 168/291 baseline ran on the old engine. The harness now has a `qwen2.5-7b-awq` candidate on vLLM v0.30.0; run it before relying on `model: qwen`.
- **Thinking** is per ModelConfig: `flo-llm-think` sets `reasoningEffort: low`, which vLLM maps to Gemma's thinking switch. Under the `qwen` profile the field is ignored, so the toggle is harmless.

## 4. Chart versions

| Chart | Live / upstream | This PR | Why |
|---|---|---|---|
| `docs-agent` | 0.1.0 | **0.3.0** | The cherry-picked live-config commit used 0.2.0, and the observability branch uses 0.2.0 for different content. 0.3.0 keeps versions moving forward and avoids the collision. An earlier draft set it back to 0.2.0; that was the "downgrade" and it is reverted |
| `llm-runtime` (was `qwen-runtime`) | 0.1.0 (from #240), never installed | 0.2.0 | Breaking values change (catalog); no live release depends on 0.1.0 |
| `gateway-guardrails` | live 0.1.0 (release rev 4); 0.1.1 in this branch's history (#240) | 0.2.0 | Breaking values change: `qwenInferenceServiceName` became the list `llmInferenceServiceNames` |

## 5. Mesh access (`gateway-guardrails`)

- `meshPolicies.llmInferenceServiceNames` (default `[llm]`) renders one `allow-kagent-to-<name>` AuthorizationPolicy per InferenceService.
- Live today, the Helm-owned `allow-kagent-to-llm` selects `qwen-llm-standard`. Upgrading with the default re-points it to the new `llm` ISVC. During the cutover, upgrade with both names so the debug agent's path to Qwen stays allowed:
  `--set-json 'meshPolicies.llmInferenceServiceNames=["llm","qwen-llm-standard"]'`.
- Impact today is nil: the LLM pods have no Istio sidecar and no namespace is injected or ambient, so these policies are not enforced on them yet. The list keeps the policy correct for when they are.
- `gateway-guardrails` is upgraded manually (CI only lints it).

## 6. CI (`.github/workflows/oke-cicd.yaml`, `tests.yml`)

- `deploy_kserve` and `deploy_gemma` are replaced by one input, `deploy_llm`. It runs `helm upgrade --install llm-runtime … --atomic --cleanup-on-fail`, waits for `deployment/<isvc>-predictor`, then runs `helm test`. A push installs `llm-runtime` only when the release is missing; otherwise pushes to main never restart the model.
- Lint and render loop over every catalog key (`yq '.catalog | keys'`), so a broken rollback profile fails CI.
- "Smoke test LLM" reads the Service name and served alias from the chart values; there are no model names in the workflow.
- The legacy Knative `qwen-llm` migration and recovery block is deleted. No `qwen-llm` objects remain on the cluster.

## 7. Tool-choice proxy: out of this PR, kept for later

- Removed: `templates/tool-choice-proxy.yaml`, `files/tool-choice-proxy.py`, `values.toolChoiceProxy`, and the proxy branch in `templates/kagent.yaml`.
- Evidence for `tool_choice: auto`: Gemma's proxy and auto runs scored identically (258/258 thinking off, 270/270 on) because vLLM ignores `tool_choice: required` for `gemma4`. The proxy cut `no_tool` for every other candidate. The debug agent scored 51/60 without it.
- **Later iteration:** the source is preserved in commit `5d61472` (`feat/flo-gemma`) and `f499f03` (`feature/observability`). Bring it back only with an eval showing it helps the active model.

## 8. Langfuse: none in this PR

There were never secret values in the PR, only `optional: true` references to the `mcp-langfuse-keys` Secret. These are removed anyway because this PR is not about observability: the `mcp.langfuse` values, the `LANGFUSE_*` env in `templates/deployment.yaml`, and the README paragraph. They move to the observability PR.

## 9. Evals: one markdown file

- `tests/eval/SLM_EVALS.md` is the single record: method, gates, full matrix, per-group table, speed, kagent end-to-end, known issues, scorer changes, iteration log.
- `tests/eval/slm_results/` is gitignored (the JSONs stay local), and `slm_chat.py` is deleted.
- `slm_tool_eval.py` and `flo_eval_dataset.json` stay, so every number is reproducible.
- `prompt_lab.py` A/B-tests docs prompts against the live model and MCP; candidate prompts are passed as file paths and are not committed.

## 10. Docs prompt

The model evals (`SLM_EVALS.md`) ran on the previous prompt. Its weaknesses on Gemma were a greeting rule that over-applied ("Sup." for off-topic questions) and no empty-results rule (blank replies). `files/docs-system-message.txt` is rewritten to fix both: identity questions are answered as Flo, off-topic questions are declined, and a missing answer is stated as not found. `prompt_lab.py` compares prompts against the live model and MCP, including whether definition questions retrieve the defining page.

## 11. Thinking as a UI toggle

- vLLM v0.30.0 (checked 2026-10-03): with `reasoning_effort` unset or `none`, thinking is off; with `low`, `medium` or `high`, it is on. kagent exposes this as `ModelConfig.spec.openAI.reasoningEffort`.
- The chart adds agent `kubeflow-docs-agent-think` on `flo-llm-think` (`maxTokens: 8192`; 1024 truncates the reasoning).
- The frontend's "Think deeper" pill routes the docs persona to that agent. Toggling starts a new chat, because kagent sessions are per agent. The pill is disabled for the debug persona.
- Expected effect: 258 → 270/291, with median reply time going from 1.3 s to 2.9 s.
- Rejected: passing a flag in A2A metadata, because kagent does not map it onto model parameters.

## Rollout

Order matters, because each step depends on the one before it:

1. `llm-runtime` first (`model: gemma`), then `helm test llm-runtime`. CI installs it automatically when the release is missing; after that only a `deploy_llm` dispatch restarts the model.
2. `docs-agent`: its ModelConfigs call `llm-stable`, so step 1 must be Ready.
3. The chat widget that sends session tokens must be live before `gateway-guardrails` runs with `sessionAuth.enforce: true`; otherwise public chat returns 403.
4. `gateway-guardrails` (Terraform-managed): apply with `llmInferenceServiceNames: [llm]` and enforcement on.

Found on the cluster and fixed in the chart:
- An empty CPU limit lets KServe inject its default (`cpuLimit: "1"`), which rejects the 4-CPU request; the server dry-run cannot see this. Every profile now sets all four CPU/memory values, and the template `required`s them.
- CRI-O on the nodes enforces fully qualified image names, so short names (`curlimages/curl`, `busybox`) fail as ambiguous. Test pods and the CI smoke test use `docker.io/...`.

Objects left from the old setup (Qwen ISVC `qwen-llm-standard`, the `llm-runtime` ServingRuntime, `qwen-llm-stable`, PVC `qwen-hf-cache`, `tool-choice-proxy`, and the ModelConfigs `slm-gemma` and `local-qwen`) are not managed by these charts and can be deleted after a stable period.

**Rollback:** set `model: qwen` and dispatch `deploy_llm`, or `helm rollback llm-runtime`; clients never change.

## Later iterations (not this PR)

- Tool-choice proxy (section 7), only with eval evidence.
- Retrieval for definition questions: hybrid search fetches only `top_k` per leg, so "what is Kubeflow" can miss the defining page (MCP server change).
- Observability PR: Langfuse, OTel, and LLM metrics scraping for the `llm` ISVC, rebased on main after this PR.
