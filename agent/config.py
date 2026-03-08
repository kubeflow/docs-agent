"""Centralized configuration for the docs-agent."""

import os

# KServe Config
KSERVE_URL = os.getenv(
    "KSERVE_URL",
    "http://llama.santhosh.svc.cluster.local/openai/v1/chat/completions"
)
MODEL = os.getenv("MODEL", "llama3.1-8B")
PORT = int(os.getenv("PORT", "8000"))

# Milvus Config
MILVUS_HOST = os.getenv(
    "MILVUS_HOST",
    "my-release-milvus.santhosh.svc.cluster.local"
)
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "docs_rag")
MILVUS_VECTOR_FIELD = os.getenv("MILVUS_VECTOR_FIELD", "vector")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2"
)
