"""Agentic RAG router for the Kubeflow Documentation AI Assistant.

This package implements a LangGraph-based agentic retrieval pipeline that
classifies user intent, dispatches to specialised retrieval strategies,
evaluates context quality, and conditionally re-retrieves before synthesis.

Graph topology::

    classify ──► retrieve ──► evaluate ──► synthesize
                                │
                                └──► re-retrieve ──► synthesize

Nodes:
    classify     – Determine intent (kubeflow_docs | platform_arch | general)
    retrieve     – Execute the appropriate Milvus search strategy
    evaluate     – Score retrieved context for relevance / sufficiency
    re-retrieve  – Broaden or refine the query and retry retrieval
    synthesize   – Build the final LLM response with citations
"""
