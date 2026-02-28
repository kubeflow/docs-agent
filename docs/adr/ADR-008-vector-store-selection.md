# ADR-008: Vector Store Selection for docs-agent

**Status:** Proposed  
**Date:** February 28, 2026  
**Context:** PR #58 merged a Feast POC alongside the existing PyMilvus path.

## Context

Recent contributions (specifically PR #58) introduced a new Feast-based feature store path (`store_via_feast()`) side-by-side with the original `store_milvus()` pipeline. Currently, both pipelines co-exist within the repository structure, but the core servers (`server/` and `server-https/`) explicitly rely only on `pymilvus` via the `milvus_search()` function. This implicit decision—running dual ingestion frameworks without a declared standard—was never formally documented, creating ambiguity for future GSoC contributors attempting to extend the architecture. Downstream issues require a canonical storage path to avoid architectural divergence and redundant effort.

## Decision

We formally adopt **MilvusClient (via `pymilvus`)** as the singular canonical vector storage path for Kubeflow `docs-agent`.

The Feast pipeline module located in `kagent-feast-mcp/pipelines/` will be maintained strictly in a "legacy" capacity for reference and experimental use cases.

## Rationale

1. **Proven & Stable:** The current Milvus stack powers the entirety of the primary `docs-agent` implementation.
2. **Lean Dependencies:** Maintaining `feast[milvus]` imposes heavy dependencies that conflict with the goal of container footprint reduction (as seen in ADR-004).
3. **Execution Velocity:** `pymilvus` provides robust thread-safe connection handling (e.g., pooling integrations) vital for the FastAPI scale goals immediately on the roadmap, without the overhead of debugging an experimental integration path.
4. **General Availability:** `feast[milvus]` is currently an Alpha integration that cannot yet serve as a robust foundation for a production-grade Agentic RAG reference architecture.

## Consequences

- Contributors will build primary database search extensions, connection pools, and retrieval logic strictly against `pymilvus` inside the `shared/` layer.
- The `kagent-feast-mcp/` pipeline artifacts are to be explicitly commented as legacy experimental components. Downstream updates are not expected to cross-port there.
- **Future Path:** We will re-evaluate Feast as an offline feature store overlay once `feast[milvus]` hits GA and the core architecture is significantly hardened.
