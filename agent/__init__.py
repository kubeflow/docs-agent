"""Kagent-native MCP server for the Kubeflow Documentation AI Assistant.

This package implements a partition-aware MCP (Model Context Protocol) server
that the Kagent Agent CRD orchestrates via its system prompt.  The LLM is the
router — the Agent CRD's ``systemMessage`` tells the model when to call each
tool, and the tools handle retrieval quality internally.

Architecture::

    User ──► Kagent Agent CRD (LLM router)
                 │
                 ├──► search_kubeflow_docs  ──► _search(partition="")
                 │                                      │
                 ├──► search_platform_docs  ──► _search(partition="platform_arch")
                 │                                      │
                 └──► search_all_docs       ──► _search(partition="")
                                                        │
                                                        ▼
                                              Milvus vector DB
                                         (via shared/rag_core.py config)

Key improvements over the kagent-feast-mcp POC (#58):
    - Thread-safe encoder & connection init (no ``_init()`` race condition)
    - Partition-aware search (kubeflow_docs vs platform_arch collections)
    - Quality-aware retrieval with automatic query broadening on low relevance
    - Uses shared config (MILVUS_HOST, EMBEDDING_MODEL, etc.) from rag_core
    - Health-check-ready for k8s readiness/liveness probes
    - Structured logging with per-request context

Tools:
    search_kubeflow_docs  – Kubeflow Pipelines, KServe, Katib, Notebooks, SDK
    search_platform_docs  – Deployment, Terraform, OCI/OKE, Helm, Istio, GPUs
    search_all_docs       – Cross-cutting queries spanning both partitions
"""
