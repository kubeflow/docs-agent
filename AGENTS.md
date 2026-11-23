# AGENTS.md

Guidance for anyone changing this repo with an AI agent, and for AI reviewers
(CodeRabbit reads this file). Humans: [CONTRIBUTING.md](CONTRIBUTING.md) comes first.

## What this repo is

Flo, the Kubeflow docs assistant: an agentic RAG system on Kubernetes.

| Path | What |
| --- | --- |
| `docs-agent-mcp/mcp-server/` | FastMCP tools the agents call: search, rerank, compress, cite |
| `docs-agent-mcp/pipelines/` | Kubeflow Pipelines that ingest docs, code and issues into Milvus |
| `docs-agent-mcp/charts/` | Helm charts: `docs-agent`, `llm-runtime` (vLLM on KServe), `gateway-guardrails`, `pr-eval` |
| `docs-agent-mcp/terraform/` | OKE infrastructure |
| `deploy/argocd/` | Argo CD Applications for prod, staging and PR previews |
| `frontend/` | Chat widget embedded in kubeflow.org |
| `tests/`, `tests/eval/` | Unit tests; evaluation harness and datasets |
| `legacy/` | Archived; do not change |

## Pull requests

- Sign off every commit (`git commit -s`); DCO is required.
- Title: `<type>(<area>): <summary>`. Types: `bug`, `feat`, `chore`, `docs`,
  `test`, `ci`, `security`. Areas: `mcp`, `pipelines`, `frontend`, `kagent`,
  `gateway`, `terraform`, `embeddings`, `ci`, `tests`, `docs`, `infra`.
- Keep a PR to one change. Do not commit plans, notes or generated reports.
- Do not add AI attribution lines to commits or PR descriptions.

## Checks to run before pushing

```bash
ruff check .
ruff format --check docs-agent-mcp/mcp-server tests
pytest                     # unit tests only; see pytest.ini markers
helm lint docs-agent-mcp/charts/<chart>
```

Unit tests must not need network, Milvus, KServe or model downloads. Mark them
`integration` or `slow` otherwise.

## Rules that are easy to miss

- **Pinning.** Images use an explicit version tag (the vLLM image also a
  digest); Hugging Face models are pinned by revision. No `latest`.
- **Charts.** Bump `version` in `Chart.yaml` on any chart change. Model,
  context-length and GPU decisions are recorded in
  `docs-agent-mcp/charts/RUNTIME_DECISIONS.md`.
- **Model contract.** Agents call the LLM by its served name (`flo-llm`), never
  by the model ID, so changing the model does not change any client.
- **Production.** Prod Argo apps are pinned on lines marked `# prod-revision`.
  Only `deploy/promote.sh` changes them, in a PR of its own.
- **GPU apps.** `llm-runtime` and `llm-judge` sync manually; never add an
  automated `syncPolicy` to them. Do not shrink an existing PVC.
- **Secrets.** No credentials, tokens, OCIDs or internal IPs in values files,
  manifests or tests. Charts reference existing Secrets.
- **Evaluation.** The judge model is a different family from the generator.
  Changes to retrieval, reranking, compression, routing, prompts or citations
  change answers: say in the PR whether a maintainer should run `/eval`, and
  whether staging needs a re-ingest first.
- **Ingestion.** Pipelines are idempotent: re-runs replace a document's chunks,
  and deleted or renamed sources remove their vectors.
- **Widget.** Model output and citation URLs are untrusted. Escape or sanitize
  before inserting into the page; allow only `http(s)` links.
