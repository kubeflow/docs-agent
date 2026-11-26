# Kubeflow RAG pipelines

Kubeflow Pipelines (KFP) definitions for indexing Kubeflow documentation and related corpora into Milvus.

## Layout

| Path | Role |
|------|------|
| **`kubeflow-pipeline.py`** | Production **docs v4** pipeline: GitHub docs download → chunk/embed → Milvus store |
| **`issues-pipeline.py`** | GitHub **issues** RAG pipeline |
| **`code-pipeline.py`** | GitHub **code/manifests** RAG pipeline |
| **`utils/`** | Shared helpers used by the live pipelines: ingest, parsers, Milvus store, TEI |
| **`ingest_schedule.py`** | Registers the pipelines as Kubeflow Pipelines recurring runs, or starts one run by hand |
| **`Dockerfile.pipeline`** | Ingest image (`docs-rag-ingest`): the docs helpers as flat `/app` modules for the chunk and store steps, plus the pipelines and `ingest_schedule.py`. CI pushes it for every commit on main, tagged with the short SHA |
| **`legacy/pipelines/`** (repo root) | Older pipelines, including **incremental** docs ingest — **not** the live v4 docs path |

## Recurring runs

Ingest runs on a schedule as Kubeflow Pipelines recurring runs. The schedule is the
`ingest.schedules` list in the `docs-agent` chart values. After every Argo CD sync of an
environment with `ingest.enabled: true` (production), a Job runs
`ingest_schedule.py register` from the ingest image of that commit: it compiles the
pipelines, uploads them as a new version of `flo-ingest-{docs,issues,code}` and replaces
the recurring runs in the `flo-ingest` experiment. Recurring runs that are no longer in
the list are removed.

| Schedule | Pipeline | When (UTC) | What |
|---|---|---|---|
| `docs-website` | docs | daily 02:00 | `kubeflow/website` `content/en/docs` |
| `docs-distribution` | docs | daily 02:30 | `kubeflow/community-distribution` Markdown (README, Helm chart READMEs, releases, proposals), `doc_type: distribution` |
| `issues` | issues | every 6 h | issues updated in the last 24 h, ten repositories |
| `code` | code | Sunday 03:00 | `kubeflow/community-distribution` manifests and Helm charts, without CRDs |

Both docs schedules write to `kubeflow_docs`. Each re-downloads its whole source, replaces
the chunks of every file, and removes files that are gone upstream. If more than 20% of a
source's indexed files would be removed, the run fails before writing anything: that
is a broken download, not upstream deleting its docs.

Start a run by hand (from a laptop, port-forward `svc/ml-pipeline` in `kubeflow` first):

```bash
helm template docs-agent ../charts/docs-agent --set ingest.enabled=true \
  -s templates/ingest-schedule-job.yaml | yq 'select(.kind == "ConfigMap") | .data."schedules.json"' > schedules.json
export DOCS_INGEST_IMAGE=ghcr.io/kubeflow/docs-rag-ingest:<short sha>
python ingest_schedule.py run --config schedules.json --host http://localhost:8888 --name docs-distribution
# Into a test copy of the docs collection (any name starting with kubeflow_docs_):
python ingest_schedule.py run --config schedules.json --host http://localhost:8888 \
  --name docs-distribution --set collection_name=kubeflow_docs_staging
# First backfill of issues (no time window):
python ingest_schedule.py run --config schedules.json --host http://localhost:8888 \
  --name issues --set updated_within_hours=0
```

The pipelines need these Secrets in the Kubeflow Pipelines namespace: `github-pat`
(`Github_Pat`), `milvus-auth` (`MILVUS_USER`, `MILVUS_PASSWORD`) and
`milvus-maintenance-lock`.

## Docs pipeline (core)

**Purpose:** Sync one documentation source (a directory of Markdown in a GitHub repository) into the v4 hybrid Milvus collection (`kubeflow_docs`).

1. **Download GitHub directory** — every `.md` / `.html` under a repo path at the default branch head; any unreadable file fails the run (stock Python image)  
2. **Chunk and embed** — thin step that imports `hugo_ingest` / `canonical_rag_ingest` from the ingest image, then POSTs to TEI  
3. **Store in Milvus** — thin step that imports `milvus_store` helpers from the same image

Compile writes the YAML recipe (no Docker required). A **cluster run** needs the ingest image:

```bash
cd docs-agent-mcp/pipelines
# The chunk and store steps run in this image; CI tags it with the short commit SHA.
export DOCS_INGEST_IMAGE=ghcr.io/kubeflow/docs-rag-ingest:<short sha>
python kubeflow-pipeline.py
```

Key defaults: `target_tokens=350`, `overlap_tokens=50`, 768-d dense vectors, explicit `clean_rebuild` confirmation for destructive drops.

## Issues and code pipelines

Compile from the pipelines directory (same as CI):

```bash
python issues-pipeline.py
python code-pipeline.py
```

These pipelines are self-contained KFP components. Parsers and test helpers live in `utils/issues_utils.py` and `utils/code_utils.py`.

## Incremental docs ingest

The incremental GitHub docs pipeline lives under **`legacy/pipelines/incremental-pipeline.py`**. It predates v4 hybrid schema and is kept for reference only; use **`kubeflow-pipeline.py`** for production docs indexing.

## Dependencies

See `requirements.txt`. Compile-time needs `kfp` (and optionally `kfp-kubernetes` for secret wiring in cluster runs).
