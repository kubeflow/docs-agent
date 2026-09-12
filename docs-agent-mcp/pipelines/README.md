# Kubeflow RAG pipelines

Kubeflow Pipelines (KFP) definitions for indexing Kubeflow documentation and related corpora into Milvus.

## Layout

| Path | Role |
|------|------|
| **`kubeflow-pipeline.py`** | Production **docs v4** pipeline: GitHub docs download → chunk/embed → Milvus store |
| **`milvus_store.py`** | Schema, safety gates, and store helpers imported by the ingest image |
| **`Dockerfile.pipeline`** | Slim ingest image (`docs-rag-ingest`) that copies the `.py` modules above |
| **`canonical_rag_ingest.py`**, **`hugo_ingest.py`**, **`utils.py`** | Ingest and embedding utilities used by the docs pipeline and tests |
| **`github_rag_pipeline.yaml`** | Compiled docs pipeline (regenerate via `python kubeflow-pipeline.py`) |
| **`extra/issues-pipeline.py`** | GitHub **issues** RAG pipeline (+ `issues_utils.py`) |
| **`extra/code-pipeline.py`** | GitHub **code/manifests** RAG pipeline (+ `code_utils.py`) |
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

## Extra pipelines

Compile from the pipelines directory (same as CI):

```bash
python extra/issues-pipeline.py
python extra/code-pipeline.py
```

These pipelines are self-contained KFP components with helpers in `extra/*_utils.py` for unit tests.

## Incremental docs ingest

The incremental GitHub docs pipeline lives under **`legacy/pipelines/incremental-pipeline.py`**. It predates v4 hybrid schema and is kept for reference only; use **`kubeflow-pipeline.py`** for production docs indexing.

## Dependencies

See `requirements.txt`. Compile-time needs `kfp` (and optionally `kfp-kubernetes` for secret wiring in cluster runs).
