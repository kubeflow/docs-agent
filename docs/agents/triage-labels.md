# Triage labels

Canonical label strings for docs-agent issue triage. Create them with
`./scripts/sync-github-labels.sh <owner/repo>` before the first analyzer run.
GitHub Agentic Workflows will not apply a label that is missing from the repo
or from the workflow `safe-outputs.add-labels.allowed` list.

Issue titles must be `<type>(<area>): <summary>`. Templates seed that format.
The analyzer treats the parsed type and area as trusted metadata; the issue
body is untrusted.

## Kind (exactly one)

| Title type | Label | Meaning |
| --- | --- | --- |
| `bug` | `kind/bug` | Something is broken in current behavior |
| `feat` | `kind/feature` | New capability |
| `chore` | `kind/chore` | Cleanup, refactor, dependency, ops |
| `docs` | `kind/docs` | Documentation only |
| `test` | `kind/chore` + `area/tests` | Test-only change |
| `ci` | `kind/chore` + `area/ci` | GitHub Actions / CD |
| `security` | `kind/security` | Auth, injection, XSS, secrets, mesh |

## Area (exactly one)

Token in the title maps to one `area/*` label. Aliases live in
[`area-map.json`](./area-map.json).

| Title area | Label | Read first |
| --- | --- | --- |
| `mcp` | `area/mcp` | `docs-agent-mcp/mcp-server/server.py` |
| `pipelines` | `area/pipelines` | `docs-agent-mcp/pipelines/` |
| `frontend` | `area/frontend` | `frontend/docs_scripts/chatbot.js` |
| `kagent` | `area/kagent` | `docs-agent-mcp/manifests/kagent/setup.yaml` |
| `gateway` | `area/gateway` | `docs-agent-mcp/charts/gateway-guardrails/` |
| `terraform` | `area/terraform` | `docs-agent-mcp/terraform/` |
| `embeddings` | `area/embeddings` | `docs-agent-mcp/mcp-server/embeddings_client.py` |
| `ci` | `area/ci` | `.github/workflows/` |
| `tests` | `area/tests` | `tests/` |
| `docs` | `area/docs` | `README.md`, `docs/agents/` |
| `infra` | `area/infra` | Milvus, KServe, Istio, namespaces |

Use `area/mcp` for FastMCP tools, retrieval bugs, MCP Deployment, and the
`/mcp` handshake. Do not invent `area/mcp-server`; that alias resolves to
`area/mcp`.

## Status and routing (optional)

| Label | When |
| --- | --- |
| `needs-triage` | Default from templates; remove once the analyzer (or a maintainer) finishes |
| `needs-info` | Missing repro, expected/actual, logs, or environment |
| `good first issue` | Scoped, documented, no cluster required |
| `help wanted` | Maintainers want external help |
| `maintainer-only` | Agentic RAG core — design discussion first |
| `gsoc-2026` | Tracked in `gsoc2026_agentic_rag.md` |
| `duplicate` | Same defect as another issue; cite `#N` in the comment |

## Priority (at most one)

| Label | When |
| --- | --- |
| `priority/p0` | Security incident, data loss, or public-bot outage |
| `priority/p1` | Major regression or blocker with no workaround |
| `priority/p2` | Normal actionable work |

Prefer leaving priority unset over guessing.
