"""Configuration constants for the Kagent-native MCP server.

All values are overridable via environment variables so that the same
image can be used across dev / staging / prod without code changes.

In the Kagent architecture the LLM handles intent routing through the
Agent CRD's ``systemMessage``, so keyword lists and LLM-specific
settings live in the CRD manifests, not here.  This module only
contains retrieval-layer config that the MCP tools consume.
"""

import os
from typing import Dict, List


# ---------------------------------------------------------------------------
# Intent categories (used for partition mapping)
# ---------------------------------------------------------------------------


class Intent:
    """Enum-like constants for documentation partition categories."""

    KUBEFLOW_DOCS = "kubeflow_docs"
    PLATFORM_ARCH = "platform_arch"
    GENERAL = "general"

    ALL: List[str] = [KUBEFLOW_DOCS, PLATFORM_ARCH, GENERAL]


# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------

#: Default top-k for Milvus vector search.
DEFAULT_TOP_K: int = int(os.getenv("AGENT_DEFAULT_TOP_K", "5"))

#: Minimum average similarity score (0–1) to consider context "sufficient".
#: Below this threshold the tool will broaden the query and retry.
RELEVANCE_THRESHOLD: float = float(
    os.getenv("AGENT_RELEVANCE_THRESHOLD", "0.35")
)

#: Maximum number of search attempts before returning whatever results
#: are available.  The first attempt uses the original query; subsequent
#: attempts broaden it.
MAX_RETRIEVAL_ATTEMPTS: int = int(
    os.getenv("AGENT_MAX_RETRIEVAL_ATTEMPTS", "2")
)

# ---------------------------------------------------------------------------
# Milvus partition mapping
# ---------------------------------------------------------------------------

#: Maps intent categories to Milvus collection partition names.
#: An empty string means "search the full collection" (no partition filter).
#:
#: The ``platform_arch`` partition is populated by the ingestion pipeline
#: defined in ``pipelines/platform-architecture-pipeline.py``.
PARTITION_MAP: Dict[str, str] = {
    Intent.KUBEFLOW_DOCS: os.getenv("MILVUS_PARTITION_DOCS", ""),
    Intent.PLATFORM_ARCH: os.getenv(
        "MILVUS_PARTITION_PLATFORM", "platform_arch"
    ),
}
