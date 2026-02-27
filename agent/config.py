"""Configuration constants for the agentic RAG router.

All values are overridable via environment variables so that the same
image can be used across dev / staging / prod without code changes.
"""

import os
from typing import Dict, List


# ---------------------------------------------------------------------------
# Intent categories
# ---------------------------------------------------------------------------

class Intent:
    """Enum-like constants for user-intent categories."""

    KUBEFLOW_DOCS = "kubeflow_docs"
    PLATFORM_ARCH = "platform_arch"
    GENERAL = "general"

    ALL: List[str] = [KUBEFLOW_DOCS, PLATFORM_ARCH, GENERAL]


# ---------------------------------------------------------------------------
# Keyword-based intent classification signals
# ---------------------------------------------------------------------------

#: Keywords that strongly suggest the user is asking about Kubeflow
#: documentation topics (Pipelines, KServe, Katib, Notebooks, SDK, etc.).
KUBEFLOW_KEYWORDS: List[str] = [
    "kubeflow",
    "kfp",
    "pipeline",
    "pipelines",
    "kserve",
    "inferenceservice",
    "katib",
    "notebook",
    "jupyter",
    "sdk",
    "training",
    "tfjob",
    "pytorchjob",
    "mpijob",
    "xgboostjob",
    "paddlejob",
    "experiment",
    "trial",
    "suggestion",
    "profile",
    "manifest",
]

#: Keywords that suggest questions about platform-level architecture,
#: deployment, infrastructure, or the Kubernetes layer beneath Kubeflow.
PLATFORM_KEYWORDS: List[str] = [
    "terraform",
    "oci",
    "oke",
    "oracle",
    "deploy",
    "deployment",
    "infrastructure",
    "cluster",
    "helm",
    "istio",
    "knative",
    "cert-manager",
    "dex",
    "oidc",
    "oauth",
    "ingress",
    "load balancer",
    "gpu",
    "node pool",
    "autoscal",
    "kustomize",
    "argocd",
    "gitops",
]

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------

#: Default top-k for Milvus vector search.
DEFAULT_TOP_K: int = int(os.getenv("AGENT_DEFAULT_TOP_K", "5"))

#: Minimum average similarity score (0-1) to consider context "sufficient".
#: Below this threshold the evaluate node will trigger a re-retrieve.
RELEVANCE_THRESHOLD: float = float(
    os.getenv("AGENT_RELEVANCE_THRESHOLD", "0.35")
)

#: Maximum number of retrieve attempts before giving up and synthesising
#: with whatever context is available.
MAX_RETRIEVAL_ATTEMPTS: int = int(
    os.getenv("AGENT_MAX_RETRIEVAL_ATTEMPTS", "2")
)

# ---------------------------------------------------------------------------
# Milvus partition mapping
# ---------------------------------------------------------------------------

#: Maps intent categories to Milvus collection partition names.
#: The platform_arch partition is populated by the platform architecture
#: ingestion pipeline (see ``pipelines/platform-architecture-pipeline.py``).
PARTITION_MAP: Dict[str, str] = {
    Intent.KUBEFLOW_DOCS: os.getenv("MILVUS_PARTITION_DOCS", ""),
    Intent.PLATFORM_ARCH: os.getenv("MILVUS_PARTITION_PLATFORM", "platform_arch"),
}

# ---------------------------------------------------------------------------
# LLM settings for the agent
# ---------------------------------------------------------------------------

AGENT_MODEL: str = os.getenv("AGENT_MODEL", os.getenv("MODEL", "llama3.1-8B"))
AGENT_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
