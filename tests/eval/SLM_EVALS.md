# Flo small-model evals

Tracks which LLM powers Flo and why. This file is the record: raw run outputs stay local (`slm_results/`, gitignored). Add a row to the iteration log and a results section whenever a model, prompt, parser, or serving flag changes, and keep the old sections so the next iteration has the full history.

## Current state (2026-10-04)

One model serves every agent. It is picked by one value (`model:`) in the `llm-runtime` chart and always served under the alias `flo-llm` at `http://llm-stable.ml-infra.svc.cluster.local/v1`, so agents and clients never change when the model does.

| Agent | ModelConfig | Thinking |
|---|---|---|
| `kubeflow-docs-agent` (Flo) | `flo-llm` | off (server default) |
| `kubeflow-docs-agent-think` (UI "Think deeper" toggle) | `flo-llm-think` (`reasoningEffort: low`, `maxTokens: 4096`) | on |
| `kubeflow-debug-agent` | `flo-llm` | off |

| `llm-runtime` profile | Model | Role |
|---|---|---|
| `gemma` (default) | `google/gemma-4-E4B-it@ee0ef602` | active |
| `qwen` | `Qwen/Qwen2.5-7B-Instruct-AWQ@b2503754` | one-value rollback (`model: qwen`) |

Both profiles run on the same vLLM v0.30.0 image. Every agent calls the model directly with `tool_choice: auto`; there is no tool-choice proxy (see "Tool-choice proxy" below). The debug agent moves to Gemma because Gemma think-off matches Qwen on the debug cases (51/60 each).

Live since the 2026-10-04 cutover (`docs-agent-mcp/charts/RUNTIME_DECISIONS.md`): the `llm-runtime` release serves `gemma` on one A10, and the other A10 is free. Before the cutover, every round-2 run below re-checked both profiles on the release itself: Gemma reproduced its round-1 scores exactly, and Qwen on vLLM v0.30.0 matched its old-engine baseline, so `model: qwen` is a validated rollback.

Why Gemma: Flo answers in about 1.3 s median instead of about 4 s, scores 258/291 against 168/291 for the current model on the same harness, and routes every docs, issues, and query-fidelity case correctly. Thinking on scores higher (270/291) and is the only Gemma setup that passes every gate below, at about 2.9 s median.

Why the thinking toggle: Kagent cannot send `chat_template_kwargs`, but vLLM v0.30.0 maps `reasoning_effort` onto Gemma's thinking switch (checked 2026-10-03: unset or `none` = off; `low`, `medium`, `high` = on). The kagent ModelConfig field `reasoningEffort` therefore selects thinking per agent, and the frontend switches agents.

## What the harness measures

`slm_tool_eval.py` calls the model's OpenAI-compatible endpoint directly (no Kagent, no MCP) with Flo's real system prompts and MCP tool schemas plus Kagent's `ask_user`. Tool results are canned fixtures rendered by `mcp-server/citations.py`, so the model sees prod formatting. It scores only the next assistant turn.

Dataset: `flo_eval_dataset.json`, 97 cases, run 3 times each at temperature 0 (291 scored turns).

| Group | Cases | Checks |
|---|---|---|
| `docs_routing` | 15 | Calls `search_kubeflow_docs` for docs questions |
| `query_fidelity` | 10 | Search query keeps versions, component names, error strings |
| `issues_routing` | 10 | Error traces go to `search_github_issues` |
| `no_tool` | 12 | Greetings and off-topic: no tool call, on-topic reply, no prompt leak |
| `grounded_answer`, `dates_latest`, `granular_facts`, `issues_answer` | 15 | Answers from tool results; every number must appear in the evidence |
| `version_trap`, `empty_or_wrong` | 4 | Says "not found" instead of substituting another version or topic |
| `injection` | 2 | Ignores instructions and links planted in tool results |
| `multi_turn` | 10 | Follow-ups, thanks, topic switches |
| `debug_routing`, `debug_grounding` | 19 | Debug agent: right tool, copies YAML only from evidence, never merges manifests, no file paths |

Answer checks: no URLs, no `[cN]` labels, sentence limit (docs), no paths (debug), required and forbidden phrases, grounded numbers.

### Gates and decision rule (fixed before any candidate ran)

1. Must pass: 0 tool-call parse errors; at least 32/36 `no_tool` turns; 0 answers with URLs; beat the baseline overall.
2. Rank passing setups by overall score.
3. Within 2 cases, the faster single-stream decode wins.

## Results: round 1 (2026-10-02 to 2026-10-03)

Hardware: NVIDIA A10 24 GB. Serving: `docker.io/vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90` (v0.30.0), bf16, `--max-model-len 32768 --gpu-memory-utilization 0.90 --enable-auto-tool-choice`. Thinking-on runs use `max_tokens` 4096, thinking-off 1024. "Proxy" rows applied the tool-choice proxy that ran in prod at the time (chit-chat → `none`, other user turns → `required`); the proxy and the harness's proxy mode have since been removed.

| Model | Revision | Parser flags |
|---|---|---|
| Qwen2.5-7B-Instruct-AWQ (baseline, prod) | KServe huggingfaceserver v0.15.2 / vLLM 0.8.5 | `--tool-call-parser hermes` |
| `ibm-granite/granite-4.2-3b` | `e459acceac81e5fe67c07d9cfc72329a332e7eb1` | `--tool-call-parser qwen3_coder` |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `--tool-call-parser qwen3_coder --reasoning-parser qwen3` |
| `google/gemma-4-E4B-it` | `ee0ef6023621cff504d758262d4e04895a5af4a2` | `--tool-call-parser gemma4 --reasoning-parser gemma4` |

### Full matrix

| Run | Overall | Docs | Debug | Call | Answer | No-call | `no_tool` | URLs | Gates | p50 s | p95 s | Tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B, prod, no proxy | 168 | 117/231 | 51/60 | 45/153 | 81/93 | 42/45 | 33/36 | 3 | baseline | 4.07 | 15.26 | 44.6 |
| Granite, think off, auto | 215 | 185/231 | 30/60 | 95/153 | 75/93 | 45/45 | 36/36 | 0 | pass | 1.67 | 2.79 | 59.2 |
| Granite, think off, proxy | 216 | 183/231 | 33/60 | 114/153 | 75/93 | 27/45 | 24/36 | 0 | fail | 1.64 | 2.11 | 52.4 |
| Granite, think on, auto | 228 | 192/231 | 36/60 | 108/153 | 78/93 | 42/45 | 33/36 | 0 | pass | 3.48 | 8.19 | 251.3 |
| Granite, think on, proxy | 225 | 186/231 | 39/60 | 120/153 | 78/93 | 27/45 | 24/36 | 0 | fail | 1.88 | 5.87 | 140.6 |
| Qwen3.5-4B, think off, auto | 258 | 225/231 | 33/60 | 147/153 | 66/93 | 45/45 | 36/36 | 0 | pass | 1.70 | 2.74 | 45.4 |
| Qwen3.5-4B, think off, proxy | 240 | 207/231 | 33/60 | 147/153 | 66/93 | 27/45 | 24/36 | 0 | fail | 1.71 | 2.75 | 46.9 |
| Qwen3.5-4B, think on, auto | 261 | 219/231 | 42/60 | 147/153 | 72/93 | 42/45 | 36/36 | 3 | fail | 2.93 | 5.21 | 121.5 |
| Qwen3.5-4B, think on, proxy | 246 | 204/231 | 42/60 | 147/153 | 72/93 | 27/45 | 24/36 | 3 | fail | 3.02 | 5.22 | 123.5 |
| **Gemma 4 E4B, think off, auto (deployed)** | **258** | 207/231 | 51/60 | 150/153 | 75/93 | 33/45 | 27/36 | 0 | fail (`no_tool`) | **1.34** | 2.81 | 33.3 |
| Gemma 4 E4B, think off, proxy | 258 | 207/231 | 51/60 | 150/153 | 75/93 | 33/45 | 27/36 | 0 | fail | 1.34 | 2.80 | 33.3 |
| **Gemma 4 E4B, think on, auto (gate winner)** | **270** | 219/231 | 51/60 | 147/153 | 84/93 | 39/45 | 33/36 | 0 | pass | 2.86 | 8.29 | 126.0 |
| Gemma 4 E4B, think on, proxy | 270 | 219/231 | 51/60 | 147/153 | 84/93 | 39/45 | 33/36 | 0 | pass | 2.86 | 8.08 | 126.3 |

All runs: 0 tool-call parse errors, 0 request errors. Every candidate run was deterministic across the 3 repeats except Granite think-off auto (`iss-ui-crashloop`) and Gemma think-on (`dbg-ans-missing-field`, `dbg-ans-no-combine`).

Gemma's proxy and auto rows are identical because vLLM v0.30.0 ignores `tool_choice: "required"` for the `gemma4` parser: the response reports `finish_reason: tool_calls` with an empty `tool_calls` list and a plain-text answer.

### Tool-choice proxy

Removed for now; all agents use `tool_choice: auto`. Evidence: the proxy never helped Gemma (identical scores above), it hurt `no_tool` for every other candidate (24/36 vs 33–36/36 on auto), and the debug agent on Qwen2.5-7B scores 51/60 on debug cases without it. The proxy source is preserved in git history (commit `5d61472`, `files/tool-choice-proxy.py`) for a later iteration; bring it back only with a run that shows it helps the model in use.

### Groups for the best setup of each model

| Group | Qwen2.5-7B prod | Granite think on | Qwen3.5-4B think off | Gemma think off | Gemma think on |
|---|---|---|---|---|---|
| `query_fidelity` | 0/30 | 27/30 | 30/30 | 30/30 | 30/30 |
| `docs_routing` | 9/45 | 45/45 | 45/45 | 45/45 | 45/45 |
| `issues_routing` | 6/30 | 3/30 | 30/30 | 30/30 | 30/30 |
| `multi_turn` | 18/30 | 24/30 | 27/30 | 24/30 | 27/30 |
| `no_tool` | 33/36 | 33/36 | 36/36 | 27/36 | 33/36 |
| `injection` | 3/6 | 3/6 | 3/6 | 6/6 | 6/6 |
| `version_trap` | 0/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| `empty_or_wrong` | 6/9 | 9/9 | 9/9 | 3/9 | 6/9 |
| `debug_routing` | 24/30 | 18/30 | 24/30 | 30/30 | 24/30 |
| `debug_grounding` | 24/27 | 18/27 | 9/27 | 18/27 | 24/27 |
| dates, latest, granular facts, grounded, issue answers | 45/45 | 45/45 | 45/45 | 45/45 | 45/45 |

### Speed (thinking off, `slm_tool_eval.py speed`)

Streamed docs answer: Flo system prompt, tools, a ~4.6K-token search result, 256 output tokens (`ignore_eos`). Median of 5 single-stream runs; 8 requests at concurrency 4.

| Model | Prompt tokens | TTFT | Decode tok/s | TTFT @4 | Per-stream tok/s @4 | Aggregate tok/s @4 |
|---|---|---|---|---|---|---|
| Granite 4.2 3B | 4689 | 0.75 s | 56.0 | 0.77 s | 50.8 | 176.7 |
| Qwen3.5-4B | 4832 | 0.78 s | 49.1 | 0.83 s | 45.6 | 159.1 |
| Gemma 4 E4B | 4559 | 0.77 s | 42.0 | 0.81 s | 40.7 | 144.5 |

Round 1 did not measure Qwen2.5-7B decode speed; round 2 below does, on the same vLLM v0.30.0 image.

### End-to-end through Kagent (real MCP, live index)

Same messages to a copy of the docs agent on Gemma (think off) and to live Flo on Qwen2.5-7B + proxy:

| Message | Gemma | Qwen2.5-7B + proxy |
|---|---|---|
| hi | 1.0 s, "Hey." | 2.4 s, greeting |
| Latest Kubeflow release | 2.4 s, docs search, 26.03; date printed as raw epoch (fixed in this PR) | 8.2 s, 26.03, March 22, 2026 |
| Parallel trials in Katib | 2.3 s, `parallelTrialCount`, default 3 | 9.1 s, same |
| `rpc error: code = Unavailable` | 4.4 s, issues search with the full error string | 25.2 s, rewrote the query to "Unavailability pipeline run"; answer contained GitHub URLs |
| Default notebook volume size | 2.6 s, not in the index | 12.4 s, not in the index |
| thanks! | 0.9 s, "You're welcome!" | 1.2 s, "hello" |
| who are you? | 0.9 s, "Hi." | 6.8 s, needless docs search, then an introduction |
| Capital of France | 1.1 s, "I do not have information about the capital of France." | 6.2 s, `ask_user` and empty reply |

### Known issues in the deployed setup (Gemma, thinking off)

- **Blank reply on empty search results.** When a tool returns "No results found", the answer is empty (`ans-docs-empty`, `ans-issues-empty`). Thinking on handles both (one by retrying the search) but returns a blank reply on the wrong-component case instead.
- **Brush-off replies.** Off-topic or identity questions sometimes get only "Sup." or "Hey." because the prompt's greeting rule over-applies (`none-france`, `none-poem`, `none-who-are-you`, `multi-offtopic-after`).
- **Version trap.** Asked about 1.9 with only 1.10 evidence, it says 1.9 is not documented and then quotes 1.10's Kubernetes range. All three candidates except Granite do this.
- **Debug answers.** When given a manifest, the debug persona sometimes searches again instead of answering (`dbg-ans-rpc`, `dbg-ans-apiversion-trap`). The overall debug score still matches Qwen (51/60); live since the debug agent moved to `flo-llm` on 2026-10-04.

### Prompt

All runs used the docs prompt that is live in prod (`docs-agent-mcp/charts/docs-agent/files/docs-system-message.txt`, short form). The longer upstream prompt it replaced was never scored on these models. The brush-off replies come from its greeting rule, and it has no rule for empty search results; A/B both prompts on Gemma before changing either.

## Results: round 2, `llm-runtime` chart (2026-10-04)

Each profile ran as a Helm release of `docs-agent-mcp/charts/llm-runtime` on its own A10 (Gemma as the production release `llm-runtime`; Qwen as a temporary `llm-eval-qwen` release with `--set model=qwen`), served as `flo-llm` and scored through `svc/<release>-stable` with the same harness, dataset, and scorer as round 1.

| Run | Overall | Docs | Debug | Call | Answer | No-call | `no_tool` | URLs | Gates | p50 s | p95 s | Tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma 4 E4B, think off (`model: gemma`, live) | 258 | 207/231 | 51/60 | 150/153 | 75/93 | 33/45 | 27/36 | 0 | fail (`no_tool`) | 1.35 | 2.83 | 32.6 |
| Gemma 4 E4B, think on (`flo-llm-think`) | 270 | 219/231 | 51/60 | 147/153 | 84/93 | 39/45 | 33/36 | 0 | pass | 2.88 | 7.33 | 124.0 |
| Qwen2.5-7B-AWQ on vLLM v0.30.0 (`model: qwen`) | 168 | 117/231 | 51/60 | 45/153 | 81/93 | 42/45 | 33/36 | 3 | baseline | 1.10 | 2.32 | 44.0 |

All runs: 0 tool-call parse errors, 0 request errors. Gemma's per-group scores are identical to round 1 for both thinking modes (think-on is flaky on the same two debug cases). Qwen's score and groups match its round-1 baseline on KServe huggingfaceserver/vLLM 0.8.5, with lower latency (p50 1.10 s vs 4.07 s).

| Model (vLLM v0.30.0) | Prompt tokens | TTFT | Decode tok/s | TTFT @4 | Per-stream tok/s @4 | Aggregate tok/s @4 |
|---|---|---|---|---|---|---|
| Gemma 4 E4B (bf16) | 4559 | 0.78 s | 42.4 | 0.83 s | 41.1 | 145.4 |
| Qwen2.5-7B-Instruct (AWQ 4-bit) | 4618 | 0.75 s | 87.2 | 0.78 s | 80.7 | 259.2 |

Qwen decodes about twice as fast (4-bit weights), but Gemma answers in fewer tokens and is the only one that routes docs, issues, and query-fidelity cases correctly, so Gemma stays the active profile.

## Scorer changes

Results above use the current scorer. `rescore` re-applies it to saved replies without calling the model:

- Not-found detection is a regex (`NOTFOUND_RE`) instead of a phrase list; it missed correct replies such as "was found in the indexed sources" and "does not contain".
- `no_call` cases can carry `must_any`; before this, "Sup." passed "What's the capital of France?".
- `ans-version-trap-1.9` forbids quoting 1.10's range; `ans-wrong-component` forbids padding with the unrelated Katib hit.
- Replies containing raw `<tool_call>` / `<function=` text count as parse errors and fail.

## Reproduce

```bash
# Candidate on the spare GPU (Deployment, PVC, Service in slm-eval)
python tests/eval/slm_tool_eval.py manifest gemma-4-e4b | kubectl apply -f -
kubectl -n slm-eval port-forward svc/slm 8000:8000

python tests/eval/slm_tool_eval.py run --name gemma-4-e4b-thinkoff-auto --model gemma-4-e4b \
  --thinking off --repeats 3
python tests/eval/slm_tool_eval.py run --name gemma-4-e4b-thinkon-auto --model gemma-4-e4b \
  --thinking on --max-tokens 4096 --repeats 3
python tests/eval/slm_tool_eval.py speed --name gemma-4-e4b --model gemma-4-e4b --thinking off

# Qwen rollback profile on the same vLLM v0.30.0 image (validate before relying on it)
python tests/eval/slm_tool_eval.py manifest qwen2.5-7b-awq | kubectl apply -f -
python tests/eval/slm_tool_eval.py run --name qwen2.5-7b-awq-vllm030 --model qwen2.5-7b-awq --repeats 3

# Whatever llm-runtime serves now (read-only)
kubectl -n ml-infra port-forward svc/llm-stable 8081:80
python tests/eval/slm_tool_eval.py run --name flo-llm --base-url http://localhost:8081/v1 \
  --model flo-llm --repeats 3
# After changing scorer rules or the dataset's expectations
python tests/eval/slm_tool_eval.py rescore --name gemma-4-e4b-thinkoff-auto
```

Runs write `slm_results/<name>.json` (summary plus every reply) and `<name>-speed.json` locally; copy the numbers into this file.

## Iteration log

| Date | Change | Result |
|---|---|---|
| 2026-10-02 | v2 dataset (97 cases) and prod baseline: Qwen2.5-7B direct, no proxy | 168/291; skips tools on most Kubeflow questions |
| 2026-10-02 | Granite 4.2 3B, 4 setups | best 228/291 (think on); weak issue routing |
| 2026-10-03 | Qwen3.5-4B, 4 setups | 258/291 think off; think on leaks URLs |
| 2026-10-03 | Gemma 4 E4B, 4 setups | 270/291 think on (gate winner), 258/291 think off |
| 2026-10-03 | Flo docs agent switched to Gemma think off; debug agent unchanged | 1–4 s replies end to end vs 2–25 s |
| 2026-10-03 | `citations.py` renders `release_date` as `YYYY-MM-DD` | fixes raw epoch dates in answers |
| 2026-10-03 | Tool-choice proxy removed; thinking exposed as a UI toggle (`kubeflow-docs-agent-think`, `reasoningEffort: low`) | users pick 258/291 at ~1.3 s or 270/291 at ~2.9 s |
| 2026-10-03 | One `llm-runtime` chart, model picked by `model:` (catalog: `gemma`, `qwen`), alias `flo-llm`; all agents on `flo-llm` / `flo-llm-think`; Qwen moved to vLLM v0.30.0 | chart ready for cutover |
| 2026-10-04 | Round 2 on the `llm-runtime` release: Gemma (think off/on) and Qwen on vLLM v0.30.0 | Gemma 258 / 270 (identical to round 1); Qwen 168 (matches baseline) |
| 2026-10-04 | Cutover: all agents on `llm-runtime` (`model: gemma`); legacy Qwen and `slm-eval/slm` stopped | second A10 freed; end-to-end replies 2–3 s (docs), 8–11 s (think) |

## Next

- Fix the blank reply on empty results (prompt rule or a guard in the agent path) and re-run `empty_or_wrong`.
- A/B the live docs prompt against the longer upstream prompt on Gemma.
- After a stable period, delete the stopped legacy objects (`RUNTIME_DECISIONS.md`, cutover step 6).
- kagent's rendered config for `kubeflow-docs-agent-think` carries `reasoning_effort: low` and its replies are slower (8–11 s vs 2–3 s); confirm reasoning tokens in vLLM metrics once request logging or tracing is available.
