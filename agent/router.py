"""LangGraph-based agentic RAG router for Kubeflow docs-agent.

This module defines a :class:`langgraph.graph.StateGraph` with the
following topology::

    classify ──► retrieve ──► evaluate ──┬──► synthesize
                                         │
                                         └──► re_retrieve ──► evaluate
                                                              (loops back)

Each node is a pure function ``(AgentState) -> dict`` that returns only
the keys it wants to update, keeping the graph easy to test and extend.

Usage::

    from agent.router import build_graph

    graph = build_graph()
    result = graph.invoke({"query": "How do I create a Kubeflow pipeline?"})
    print(result["response"])
    print(result["citations"])
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, TypedDict

# Ensure the project root is importable (mirrors the pattern in server/app.py)
try:
    from shared.rag_core import (
        deduplicate_citations,
        milvus_search,
    )
except ModuleNotFoundError:
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from shared.rag_core import (  # noqa: E402
        deduplicate_citations,
        milvus_search,
    )

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    raise ImportError(
        "langgraph is required for the agentic router. "
        "Install it with: pip install langgraph"
    )

from agent.config import (
    DEFAULT_TOP_K,
    Intent,
    KUBEFLOW_KEYWORDS,
    MAX_RETRIEVAL_ATTEMPTS,
    PARTITION_MAP,
    PLATFORM_KEYWORDS,
    RELEVANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """Typed state passed through every node in the graph.

    Only the keys a node returns are merged — everything else is carried
    forward unchanged.
    """

    # Input
    query: str

    # Classification
    intent: str  # one of Intent.ALL

    # Retrieval
    context: List[Dict[str, Any]]  # raw Milvus hits
    citations: List[str]
    retrieval_attempts: int

    # Evaluation
    avg_similarity: float
    context_sufficient: bool

    # Synthesis
    response: str


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def classify(state: AgentState) -> dict:
    """Classify user intent using lightweight keyword matching.

    This avoids an extra LLM call for routing — fast, deterministic, and
    easy to extend.  A future iteration could add an LLM-based classifier
    behind a feature flag for ambiguous queries.
    """
    query_lower = state["query"].lower()

    # Check platform-architecture keywords first (more specific)
    for kw in PLATFORM_KEYWORDS:
        if kw in query_lower:
            logger.info("Intent classified as PLATFORM_ARCH (keyword: %s)", kw)
            return {"intent": Intent.PLATFORM_ARCH}

    # Check general Kubeflow documentation keywords
    for kw in KUBEFLOW_KEYWORDS:
        if kw in query_lower:
            logger.info("Intent classified as KUBEFLOW_DOCS (keyword: %s)", kw)
            return {"intent": Intent.KUBEFLOW_DOCS}

    # Default: general knowledge — answer without retrieval
    logger.info("Intent classified as GENERAL (no keyword match)")
    return {"intent": Intent.GENERAL}


def retrieve(state: AgentState) -> dict:
    """Execute Milvus semantic search appropriate for the classified intent."""
    intent = state.get("intent", Intent.GENERAL)
    attempts = state.get("retrieval_attempts", 0) + 1

    if intent == Intent.GENERAL:
        # No retrieval needed for general queries
        return {
            "context": [],
            "citations": [],
            "retrieval_attempts": attempts,
        }

    query = state["query"]
    partition = PARTITION_MAP.get(intent, "")

    logger.info(
        "Retrieving for intent=%s query=%r partition=%r (attempt %d)",
        intent,
        query,
        partition,
        attempts,
    )

    # Use the shared milvus_search helper; partition filtering is logged
    # above but not yet passed through — milvus_search searches the full
    # collection.  A future enhancement will add a partition_name kwarg.
    result = milvus_search(query, top_k=DEFAULT_TOP_K)
    hits = result.get("results", [])

    citations = deduplicate_citations(
        [h["citation_url"] for h in hits if h.get("citation_url")]
    )

    return {
        "context": hits,
        "citations": citations,
        "retrieval_attempts": attempts,
    }


def evaluate(state: AgentState) -> dict:
    """Score retrieved context and decide whether to re-retrieve.

    The evaluation uses average similarity of the top-k results as a
    proxy for relevance.  If the score falls below
    :data:`RELEVANCE_THRESHOLD` and we haven't exhausted retry attempts,
    the graph edges to ``re_retrieve`` instead of ``synthesize``.
    """
    context = state.get("context", [])
    attempts = state.get("retrieval_attempts", 0)
    intent = state.get("intent", Intent.GENERAL)

    # General queries skip evaluation entirely
    if intent == Intent.GENERAL:
        return {"avg_similarity": 0.0, "context_sufficient": True}

    # No results for a specific intent — allow re-retrieval if attempts remain
    if not context:
        return {
            "avg_similarity": 0.0,
            "context_sufficient": attempts >= MAX_RETRIEVAL_ATTEMPTS,
        }

    avg_sim = sum(h.get("similarity", 0) for h in context) / len(context)
    sufficient = avg_sim >= RELEVANCE_THRESHOLD or attempts >= MAX_RETRIEVAL_ATTEMPTS

    logger.info(
        "Context evaluation: avg_similarity=%.3f, sufficient=%s (attempt %d/%d)",
        avg_sim,
        sufficient,
        attempts,
        MAX_RETRIEVAL_ATTEMPTS,
    )

    return {
        "avg_similarity": avg_sim,
        "context_sufficient": sufficient,
    }


def re_retrieve(state: AgentState) -> dict:
    """Broaden the query and retry retrieval.

    Strategy: append "overview setup getting started" to encourage broader
    matches.  A more sophisticated version could use the LLM to rewrite
    the query, but this keeps latency low.

    The broadened query is persisted in the returned state so that
    downstream nodes (e.g. synthesize) can reference the actual query
    used for retrieval.
    """
    original_query = state["query"]
    broadened = f"{original_query} overview setup getting started"
    logger.info("Re-retrieving with broadened query: %r", broadened)

    # Temporarily override the query for retrieval
    modified_state: AgentState = {**state, "query": broadened}  # type: ignore[typeddict-item]
    result = retrieve(modified_state)
    # Persist the broadened query so downstream nodes see what was searched
    result["query"] = broadened
    return result


def synthesize(state: AgentState) -> dict:
    """Build the final response text.

    For now this formats the context into a structured answer.  When
    integrated with the server, this node's output feeds straight into
    the streaming LLM call — the ``response`` field contains the
    assembled context block that gets prepended to the system prompt.
    """
    intent = state.get("intent", Intent.GENERAL)
    context = state.get("context", [])
    citations = state.get("citations", [])
    query = state["query"]

    if intent == Intent.GENERAL:
        # No retrieval was done — let the LLM answer from its own knowledge
        return {
            "response": (
                f"[GENERAL] No retrieval needed. "
                f"Pass query directly to LLM: {query!r}"
            ),
        }

    if not context:
        return {
            "response": (
                "I searched the Kubeflow documentation but couldn't find "
                "relevant results. Could you rephrase your question?"
            ),
        }

    # Format context for the LLM
    context_parts: List[str] = []
    for i, hit in enumerate(context, 1):
        content = hit.get("content_text", "")
        source = hit.get("file_path", "unknown")
        sim = hit.get("similarity", 0)
        context_parts.append(
            f"[{i}] (similarity={sim:.3f}, source={source})\n{content}"
        )

    context_block = "\n\n".join(context_parts)
    citation_block = "\n".join(f"- {url}" for url in citations) if citations else ""

    response = (
        f"Based on {len(context)} retrieved passages "
        f"(avg similarity: {state.get('avg_similarity', 0):.3f}):\n\n"
        f"{context_block}"
    )
    if citation_block:
        response += f"\n\nSources:\n{citation_block}"

    return {"response": response}


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------


def _should_re_retrieve(state: AgentState) -> str:
    """Return the next node name based on context sufficiency."""
    if state.get("context_sufficient", True):
        return "synthesize"
    return "re_retrieve"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Construct and compile the agentic RAG StateGraph.

    Returns
    -------
    StateGraph
        A compiled LangGraph graph ready to be invoked with
        ``graph.invoke({"query": "..."})``.
    """
    graph = StateGraph(AgentState)

    # -- Add nodes --
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("evaluate", evaluate)
    graph.add_node("re_retrieve", re_retrieve)
    graph.add_node("synthesize", synthesize)

    # -- Add edges --
    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "evaluate")

    # Conditional: evaluate → synthesize OR re_retrieve
    graph.add_conditional_edges(
        "evaluate",
        _should_re_retrieve,
        {
            "synthesize": "synthesize",
            "re_retrieve": "re_retrieve",
        },
    )

    # re_retrieve always flows to evaluate (which may then loop or proceed)
    graph.add_edge("re_retrieve", "evaluate")

    # synthesize is terminal
    graph.add_edge("synthesize", END)

    return graph.compile()
