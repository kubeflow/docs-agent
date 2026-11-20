# Kubeflow RAG pipelines

Kubeflow Pipelines (KFP) definitions for indexing Kubeflow documentation and related corpora into Milvus.

## Layout

| Path | Role |
|------|------|
| **`kubeflow-pipeline.py`** | Production **docs v4** pipeline: GitHub docs download → chunk/embed → Milvus store |
| **`issues-pipeline.py`** | GitHub **issues** RAG pipeline |
| **`code-pipeline.py`** | GitHub **code/manifests** RAG pipeline |
| **`utils/`** | Shared helpers used by the live pipelines: ingest, parsers, Milvus store, TEI |
| **`Dockerfile.pipeline`** | Slim ingest image (`docs-rag-ingest`) that copies docs helpers from `utils/` as flat `/app` modules |
| **`github_rag_pipeline.yaml`** | Compiled docs pipeline (regenerate via `python kubeflow-pipeline.py`) |
| **`legacy/pipelines/`** (repo root) | Older pipelines, including **incremental** docs ingest — **not** the live v4 docs path |

## Docs pipeline (core)

**Purpose:** Full rebuild of the documentation corpus into the v4 hybrid Milvus collection (`kubeflow_docs`).

1. **Download GitHub directory** — `.md` / `.html` under a repo path (stock Python image)  
2. **Chunk and embed** — thin step that imports `hugo_ingest` / `canonical_rag_ingest` from the ingest image, then POSTs to TEI  
3. **Store in Milvus** — thin step that imports `milvus_store` helpers from the same image

Compile writes the YAML recipe (no Docker required). A **cluster run** needs the ingest image:

```bash
cd docs-agent-mcp/pipelines
# Optional: pin a built tag (default is ghcr.io/kubeflow/docs-rag-ingest:v0.1.0)
# set DOCS_INGEST_IMAGE=ghcr.io/kubeflow/docs-rag-ingest:<tag>
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

The incremental GitHub docs pipeline (**`incremental-pipeline.py`**) processes only changed and deleted files, reusing the canonical v4 ingestion and storage utilities (`canonical_rag_ingest.py`, `milvus_store.py`) to update the v4 hybrid collection (`kubeflow_docs`) idempotently. Compile via:

```bash
python incremental-pipeline.py
```

## Dependencies

See `requirements.txt`. Compile-time needs `kfp` (and optionally `kfp-kubernetes` for secret wiring in cluster runs).
