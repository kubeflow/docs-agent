# Showcase eval queries

Three queries from the 2026 demo slides. Use them as the first regression set after any retrieval or widget change.

**Ingest first, then evaluate.** Two of the three golden sources are outside the default pipeline repo lists. Scoring the live agent against an empty hit list will look like a model bug and is not.

Collections (from `mcp-server.yaml`): `kubeflow_docs`, `issues_rag`, `code_rag`.

Do not apply pipelines to the cluster without the operator’s OK. Commands below are the intended sequence, not permission to run them.

## 1. Confirm the source is in Milvus

Exec into the MCP pod and call the tool the query is supposed to use. Every `expected_source_urls` entry in `queries.json` must appear as a `**Source:**` line.

```bash
# from docs-agent-mcp/mcp-server, or kubectl exec deploy/mcp-kubeflow-docs -n docs-agent
# search_kubeflow_docs / search_github_issues / search_kubeflow_code
# query strings and expected URLs: tests/eval/queries.json
```

If the URL is missing, **stop**. Run the ingest for that row (section 2), wait for the KFP run to succeed, then search again. Do not score the agent yet.

Known gaps against **default** pipeline arguments:

| Query | Golden source | In default ingest? |
|---|---|---|
| Katib hyperparameter tuning | `website` `content/en/docs/components/katib` → configure-experiment | Yes, if the docs full/incremental pipeline has been run on `content/en/docs` |
| KServe deploymentMode / Knative vs Serverless | `kserve/kserve#5885` | **No.** Issues default is `kubeflow/kubeflow,kubeflow/pipelines,kubeflow/manifests` |
| Katib Experiment YAML | `kubeflow/katib` `examples/v1beta1/hp-tuning/random.yaml` | **No.** Code default is `kubeflow/manifests` `apps/katib` (install YAML/CRDs) |

## 2. Ingest the missing sources

Port-forward the pipeline API if submitting from a laptop:

```bash
kubectl port-forward svc/ml-pipeline 8888:8888 -n kubeflow
cd docs-agent-mcp/pipelines
```

Docs (only if gate 1 missed the Katib page):

```python
# same pattern as submit_run.py
# directory_path="content/en/docs/components/katib"
# collection_name="kubeflow_docs"
```

Issues — **required** for query 2:

```python
client.create_run_from_pipeline_package(
    "github_issues_rag_pipeline.yaml",  # compile issues-pipeline.py if missing
    arguments={
        "repos": "kserve/kserve",
        "state": "all",
        "max_issues_per_repo": 200,
        "collection_name": "issues_rag",
        "milvus_host": "milvus-milvus.ml-infra.svc.cluster.local",
        "milvus_port": "19530",
    },
    run_name="eval-kserve-issues",
    experiment_name="docs-agent-eval",
    enable_caching=False,
)
```

Code — **required** for query 3:

```python
client.create_run_from_pipeline_package(
    "code_rag_pipeline.yaml",  # compile code-pipeline.py if missing
    arguments={
        "repos": "kubeflow/katib",
        "directory_paths": "examples/v1beta1/hp-tuning",
        "file_extensions": "yaml,yml",
        "collection_name": "code_rag",
        "milvus_host": "milvus-milvus.ml-infra.svc.cluster.local",
        "milvus_port": "19530",
    },
    run_name="eval-katib-examples",
    experiment_name="docs-agent-eval",
    enable_caching=False,
)
```

Compile if the YAML is not in the directory:

```bash
python3 issues-pipeline.py   # writes github_issues_rag_pipeline.yaml
python3 code-pipeline.py     # writes code_rag_pipeline.yaml
```

Re-run the MCP search from section 1. `curl -IL` each `expected_source_urls` entry; it must 200 (redirects OK).

## 3. Score the agent (gate 2)

The checked-in runner enforces the gate ordering and never submits ingestion:

```bash
# Read-only: exits 2 while any golden Source URL is absent.
python3 tests/eval/run_showcase_eval.py --retrieval-only

# Runs Gate 1 again, then and only then creates sessions and POSTs message/stream.
python3 tests/eval/run_showcase_eval.py

# Diagnose one row, including the exact named-tool arguments, observed tool
# Source URLs, and authoritative final answer.
python3 tests/eval/run_showcase_eval.py \
  --row-id issues-kserve-deploymentmode --verbose
```

It also reports `retrieval evidence missing` when a golden URL is present but
the returned chunks do not contain a required answer string. That diagnostic is
not a substitute for the Source-URL gate; it identifies stale/incomplete chunks
that would otherwise be misclassified as a generation failure.

Rows may define `retrieval_query` separately from the conversational `query`.
This records the focused wording the agent is expected to send to MCP and makes
query-rewrite quality independently testable from answer generation.

New chat, **Docs** persona, empty `contextId`. Widget-equivalent:

1. `POST https://agent.santhoshtoorpu.com/api/session` with `Origin: https://kubeflowdemochatbot.netlify.app`
2. `POST .../a2a/docs-agent/kubeflow-docs-agent` `message/stream`

Pass all of:

- The named `tool` appears in the stream (function-call / tool metadata).
- Every cited URL is a character-for-character `**Source:**` from gate 1. No invented `docs/components/serving/*`, no `github.com/kubeflow/kserve`.
- `must_appear_in_answer` strings are present; `forbidden_in_answer` strings are not.
- Widget: `[title](url)` is a real `<a href>` once `formatMarkdown` is fixed.

Fail the row, not the whole suite, if one query still 404s after ingest — that is an ingestion bug, not an LLM bug.

## 4. Queries

See `queries.json`. Short form:

1. `How do I configure Katib hyperparameter tuning?` → `search_kubeflow_docs`
2. `InferenceService update rejected: deploymentMode cannot be changed from Knative to Serverless` → `search_github_issues`
3. `Show me a Katib Experiment YAML example` → `search_kubeflow_code`
