# docs-agent architecture (agent reference)

This is the trusted architecture map for issue triage and coding agents.
It is the in-repo counterpart to *Agentic RAG on Kubeflow — Implementation Deep
Dive* and the GSoC spec in [`gsoc2026_agentic_rag.md`](../../gsoc2026_agentic_rag.md).

Read this file before opening source. Then open only the files listed for the
issue's `area` in [`area-map.json`](./area-map.json).

## What is running

Four-layer decoupling is live on the OCI demo cluster:

1. **Frontend** — docs-site chat widget.
2. **Middleware / agent** — kagent CRDs + FastMCP tools.
3. **Pipelines** — Kubeflow Pipelines ingestion into Milvus.
4. **Vector DB** — Milvus + a shared TEI embedding service.

Agents are Kubernetes-native (`Agent`, `ModelConfig`, `RemoteMCPServer`).
Ingestion is idempotent (delete-by-id, then insert). The public edge is one Helm
chart: TLS, 60 req/min, CORS allow-list, 30s timeout, anonymous session JWTs.

Not built: KEDA scale-to-zero, a guardrail model, a semantic router,
prompt-injection classification, a golden-dataset regression suite, and
distributed tracing.

## Namespace split

| Namespace | Owns |
| --- | --- |
| `ml-infra` | Milvus, TEI embeddings, KServe LLM |
| `docs-agent` | MCP server, kagent resources, session-issuer |
| `kubeflow` | KFP standalone |

Runtime MCP clients talk to `milvus-milvus.ml-infra.svc.cluster.local:19530`.
The LLM is `Qwen2.5-7B-Instruct-AWQ` served as `qwen2.5-7B`. That name must
match `--model_name` in the KServe manifest or every agent request 404s.

## Layer map (issue area → code)

Use the `area` token from a title `type(area): summary`. Apply the matching
`area/*` label from [`triage-labels.md`](./triage-labels.md).

### 1. Infrastructure and deployments — `area/infra`, `area/terraform`

Reference: Terraform + Helm; kagent CRDs; KEDA cron + Prometheus with
scale-to-zero; OCI budget alerts.

Current: ten Terraform files under `docs-agent-mcp/terraform/`. Namespaces are
variables. The `gateway-guardrails` chart is installed by
`gateway_guardrails.tf`.

| Concern | File |
| --- | --- |
| Namespaces | `docs-agent-mcp/terraform/namespaces.tf` |
| KServe / Knative | `docs-agent-mcp/terraform/knative.tf` |
| Milvus | `docs-agent-mcp/terraform/milvus.tf` |
| Embeddings | `docs-agent-mcp/terraform/embeddings.tf` |
| KFP | `docs-agent-mcp/terraform/kubeflow_pipelines.tf` |
| kagent | `docs-agent-mcp/terraform/kagent.tf` |
| Edge | `docs-agent-mcp/terraform/gateway_guardrails.tf` |
| LLM serve | `docs-agent-mcp/manifests/vllm/kserve-qwen.yaml` |

Gap: no KEDA ScaledObject (Issue #47). The LLM is a resident GPU pod. CD does
not redeploy KServe on every push — only when `kserve-qwen.yaml` changes.

### 2. Ingestion pipelines — `area/pipelines`

Docs, issues, and code pipelines delete by `file_unique_id` then insert. They
do **not** drop the collection. Schema version lives in the Milvus collection
description (`v=<SCHEMA_VERSION>`).

| Pipeline | Collection | File |
| --- | --- | --- |
| Docs | `docs_rag` / `kubeflow_docs` | `docs-agent-mcp/pipelines/kubeflow-pipeline.py` |
| Issues | `issues_rag` | `docs-agent-mcp/pipelines/issues-pipeline.py` |
| Code | `code_rag` | `docs-agent-mcp/pipelines/code-pipeline.py` |
| Incremental | existing records | `docs-agent-mcp/pipelines/incremental-pipeline.py` |
| Shared TEI / truncation | — | `docs-agent-mcp/pipelines/utils.py` |

Gap: chunking is bounded in characters; TEI is bounded in tokens. Dense YAML
and Go overflow (~1000 chars can still 413). Related: Issue #181.

### 3. Vector database — `area/embeddings`

One TEI InferenceService (`sentence-transformers/all-mpnet-base-v2`, 768-d,
IVF_FLAT COSINE) is shared by runtime MCP and ingestion. Three collections,
three schemas. Hybrid BM25+dense search is not enabled.

Collection strategy (one collection + partitions vs three collections) is an
open design question; settle it with the golden dataset, not intuition.

### 4. Middle agent and router — `area/mcp`, `area/kagent`

FastMCP Streamable HTTP on `:8000/mcp`. Tools are read-only.

| MCP tool | Collection | Use |
| --- | --- | --- |
| `search_kubeflow_docs` | `docs_rag` | Concepts, setup, APIs |
| `search_github_issues` | `issues_rag` | Bugs, stack traces, community fixes |
| `search_kubeflow_code` | `code_rag` | Source, YAML, resource names |

Routing is prompt-level in `docs-agent-mcp/manifests/kagent/setup.yaml`, not a
semantic router. Tools return full Markdown, not the Thin Context contract
(`source_url`, `<=150 token chunk_text`, `score`).

| Concern | File |
| --- | --- |
| Tool implementations | `docs-agent-mcp/mcp-server/server.py` |
| TEI client | `docs-agent-mcp/mcp-server/embeddings_client.py` |
| Collection names | `docs-agent-mcp/mcp-server/rag_collections.py` |
| Post-deploy smoke | `docs-agent-mcp/mcp-server/smoke_tools.py` |
| MCP Deployment | `docs-agent-mcp/manifests/mcp-server/mcp-server.yaml` |
| Agent CRDs | `docs-agent-mcp/manifests/kagent/setup.yaml` |

### 5. Frontend and edge — `area/frontend`, `area/gateway`

Widget: `frontend/docs_scripts/chatbot.js`. Public path: widget → OCI LB →
Istio gateway (TLS, rate limit, CORS, session JWT, 30s timeout) → kagent-ui →
agent → MCP + KServe.

| Concern | File |
| --- | --- |
| Chart | `docs-agent-mcp/charts/gateway-guardrails/` |
| Session JWT | `docs-agent-mcp/session-issuer/` |
| Widget | `frontend/docs_scripts/chatbot.js` |

Known edge gaps: CORS is not actually denied for off-list origins (upstream
kagent-ui still sends `*`); istiod can cache a failed JWKS fetch until restart.

## Query path (for bug reports)

1. User asks a Kubeflow question in the widget or kagent UI.
2. Request crosses the Istio gateway.
3. kagent follows the Agent system message and selects MCP tools.
4. MCP embeds the query through TEI and searches Milvus.
5. kagent sends retrieved context to the KServe LLM.
6. Answer returns with citations.

If the issue is "wrong answer", inspect kagent routing, MCP tool selection, the
collection, and TEI truncation — not only the LLM.

## Maintainer-owned vs community

Treat these as `maintainer-only` until a design issue exists:

- Semantic router / LangGraph / ADK
- MCP tool contracts and retrieval strategy
- kagent `systemMessage` routing rules
- Golden dataset / RAGAS design
- LLM gateway and quota model

Community-safe: tests, CI static checks, README drift, pipeline operational
fixes, widget UX that does not change retrieval.

## Graduation / known gaps (map incoming issues here)

| Gap | Typical area | Notes |
| --- | --- | --- |
| KEDA scale-to-zero | `infra` | Issue #47, single GPU |
| TEI 413 / token limits | `embeddings`, `pipelines` | Issue #181 |
| Thin Context MCP mode | `mcp` | Appendix C.2 of the deep dive |
| Semantic router | `kagent` | Prompt routing only |
| Guardrail model | `kagent`, `gateway` | Edge stops volume, not content |
| Golden dataset | `tests`, `frontend` | Needed for collection-strategy decision |
| Tracing | `ci`, `mcp` | Cannot attribute p95 latency |
| Chart personal defaults | `gateway` | Domain / ACME email in `values.yaml` |
| Pre-existing MCP Secret | `mcp`, `ci` | `mcp-server-secret` must exist |
| Hand-applied cluster drift | `ci` | Pipeline is not the only deploy path |
