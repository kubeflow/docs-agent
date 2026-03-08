"""Shared agent utilities for the Kubeflow Docs Assistant."""

from .config import (
    KSERVE_URL,
    MODEL,
    PORT,
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION,
    MILVUS_VECTOR_FIELD,
    EMBEDDING_MODEL,
)
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS
from .rag import milvus_search, execute_tool


