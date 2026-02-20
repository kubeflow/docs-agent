"""
agent_router.py — LangGraph Intent Router PoC for multi-source retrieval
------------------------------------------------------------------------
Resolves: https://github.com/kubeflow/docs-agent/issues/42

Problem
-------
The existing execute_tool() in server/app.py performs a single, flat Milvus
search across the entire collection.  With the upcoming Milvus partition work
(PR #12) the store will be split into three logical partitions:

  • docs      — Kubeflow website / official documentation
  • issues    — GitHub Issue text and comments
  • platform  — Architecture, Terraform & OCI deployment manifests

This file introduces a LangGraph-powered routing layer that classifies the
user's intent and dispatches to the correct partition before performing
retrieval, rather than doing a blind full-collection search every time.

Architecture
------------
                         ┌──────────────┐
  user_query  ──────────►│ intent_router │
                         └──────┬───────┘
              docs?             │issues?             platform?
              ▼                 ▼                    ▼
     retrieve_docs       retrieve_issues      retrieve_platform
              │                 │                    │
              └────────────────►▼◄───────────────────┘
                           (END / respond)

Self-correction loop
--------------------
If a retrieval node returns an empty result set the graph re-routes back to
the router with the original query so the LLM can try the next-best partition
instead of replying with "I don't know."

Usage
-----
    from server.agent_router import build_graph, AgentState

    app   = build_graph()
    state = app.invoke({"question": "How do I configure KServe auth?"})
    print(state["response"])

Environment variables (inherit from app.py)
-------------------------------------------
  KSERVE_URL          — LLM endpoint (via OpenAI-compatible API)
  MODEL               — model name
  MILVUS_HOST / PORT  — Milvus connection details
  MILVUS_COLLECTION   — base collection name
  EMBEDDING_MODEL     — sentence-transformers model id
"""

from __future__ import annotations

import os
import logging
import time
from typing import Optional

from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (mirrors app.py defaults)
# ---------------------------------------------------------------------------
KSERVE_URL = os.getenv(
    "KSERVE_URL",
    "http://llama.santhosh.svc.cluster.local/openai/v1",
)
MODEL = os.getenv("MODEL", "llama3.1-8B")
MILVUS_HOST = os.getenv("MILVUS_HOST", "my-release-milvus.santhosh.svc.cluster.local")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "docs_rag")
MILVUS_VECTOR_FIELD = os.getenv("MILVUS_VECTOR_FIELD", "vector")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
TOP_K = int(os.getenv("ROUTER_TOP_K", "5"))

# Maximum times the graph will loop back due to empty retrieval before giving up
MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Shared state passed between every node in the LangGraph."""
    question: str           # original user question (immutable)
    intent: str             # classified partition: "docs" | "issues" | "platform"
    context: str            # retrieved text from Milvus
    citations: list[str]    # source URLs collected during retrieval
    response: str           # final answer (populated at END)
    error: Optional[str]    # last tool error (enables self-correction loop)
    retries: int            # loop counter — prevents infinite recursion


# ---------------------------------------------------------------------------
# Routing schema (structured output → deterministic classification)
# ---------------------------------------------------------------------------

class RouteQuery(BaseModel):
    """Strict schema for intent classification.  The LLM *must* return one of
    the three literals — hallucinated values are rejected by Pydantic."""

    datasource: str = Field(
        ...,
        description=(
            "Route the query to exactly one partition:\n"
            "  'docs'     — conceptual / how-to questions about Kubeflow features\n"
            "  'issues'   — troubleshooting, error messages, known bugs\n"
            "  'platform' — deployment, Terraform, OCI, Kustomize, cluster setup"
        ),
    )

    def validate_datasource(self) -> str:
        allowed = {"docs", "issues", "platform"}
        if self.datasource not in allowed:
            raise ValueError(f"datasource must be one of {allowed}, got '{self.datasource}'")
        return self.datasource


# ---------------------------------------------------------------------------
# Helper: Milvus search with partition support
# ---------------------------------------------------------------------------

def _milvus_search(query: str, partition: str, top_k: int = TOP_K) -> dict:
    """Search a named Milvus partition and return hits.

    Falls back to full-collection search if the partition does not yet exist
    (e.g., during development before PR #12 is merged).
    """
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(MILVUS_COLLECTION)
        collection.load()

        encoder = SentenceTransformer(EMBEDDING_MODEL)
        query_vec = encoder.encode(query).tolist()

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}
        search_kwargs: dict = dict(
            data=[query_vec],
            anns_field=MILVUS_VECTOR_FIELD,
            param=search_params,
            limit=top_k,
            output_fields=["file_path", "content_text", "citation_url"],
        )

        # Use partition if available — graceful fallback otherwise
        if collection.has_partition(partition):
            search_kwargs["partition_names"] = [partition]
            logger.info("[ROUTER] Searching partition '%s'", partition)
        else:
            logger.warning(
                "[ROUTER] Partition '%s' not found — falling back to full collection search",
                partition,
            )

        results = collection.search(**search_kwargs)

        hits = []
        for hit in results[0]:
            similarity = 1.0 - float(hit.distance)
            entity = hit.entity
            content = entity.get("content_text") or ""
            if len(content) > 400:
                content = content[:400] + "..."
            hits.append(
                {
                    "similarity": similarity,
                    "file_path": entity.get("file_path"),
                    "citation_url": entity.get("citation_url"),
                    "content_text": content,
                }
            )
        return {"results": hits}

    except Exception as exc:
        logger.error("[ROUTER] Milvus search failed: %s", exc)
        return {"results": [], "error": str(exc)}

    finally:
        try:
            connections.disconnect(alias="default")
        except Exception:
            pass


def _format_hits(hits: list[dict]) -> tuple[str, list[str]]:
    """Convert raw Milvus hits into (formatted_text, citation_urls)."""
    citations: list[str] = []
    lines: list[str] = []
    for h in hits:
        url = h.get("citation_url", "")
        if url and url not in citations:
            citations.append(url)
        lines.append(
            f"Source: {h.get('file_path', 'unknown')}\n"
            f"Content: {h.get('content_text', '')}\n"
            f"URL: {url}\n"
            f"Score: {h.get('similarity', 0):.3f}"
        )
    text = "\n\n".join(lines) if lines else ""
    return text, citations


# ---------------------------------------------------------------------------
# Node 1 — Intent Router
# ---------------------------------------------------------------------------

def intent_router(state: AgentState) -> AgentState:
    """Classify the user's question and set state['intent']."""
    logger.info("[ROUTER] Classifying intent for: %s", state["question"][:80])

    llm = ChatOpenAI(
        base_url=KSERVE_URL,
        model=MODEL,
        temperature=0,          # deterministic routing
        max_tokens=50,          # we only need a single-word classification
    )
    structured_llm = llm.with_structured_output(RouteQuery)

    system = (
        "You are an expert routing agent for the Kubeflow documentation assistant.\n"
        "Given a user question, output ONLY a JSON object with a 'datasource' key.\n\n"
        "Rules:\n"
        "  • 'docs'     — general Kubeflow usage, features, SDK, Pipelines, KServe, Katib\n"
        "  • 'issues'   — error messages, bugs, crash logs, 'why is X failing'\n"
        "  • 'platform' — Terraform, Kubernetes manifests, OCI, Kustomize, cluster setup\n\n"
        "If unsure, prefer 'docs'."
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{question}")]
    )

    try:
        result: RouteQuery = (prompt | structured_llm).invoke(
            {"question": state["question"]}
        )
        intent = result.datasource
        # Validate — fall back to 'docs' if the LLM returns an unexpected value
        if intent not in {"docs", "issues", "platform"}:
            logger.warning("[ROUTER] Unexpected intent '%s', defaulting to 'docs'", intent)
            intent = "docs"
    except Exception as exc:
        logger.error("[ROUTER] Classification failed: %s — defaulting to 'docs'", exc)
        intent = "docs"

    logger.info("[ROUTER] Intent: %s", intent)
    return {**state, "intent": intent, "error": None}


# ---------------------------------------------------------------------------
# Node 2a/b/c — Retrieval nodes (one per partition)
# ---------------------------------------------------------------------------

def _retrieve(partition: str, state: AgentState) -> AgentState:
    """Generic retrieval executor — shared by all three partition nodes."""
    logger.info("[ROUTER] Retrieving from partition '%s'", partition)

    raw = _milvus_search(state["question"], partition)
    hits = raw.get("results", [])
    milvus_error = raw.get("error")

    if milvus_error:
        # Surface the error into state so the self-correction edge can see it
        return {**state, "context": "", "citations": [], "error": milvus_error}

    if not hits:
        logger.warning("[ROUTER] Empty results from partition '%s'", partition)
        return {
            **state,
            "context": "",
            "citations": [],
            "error": f"No results found in partition '{partition}'",
        }

    context, citations = _format_hits(hits)
    return {**state, "context": context, "citations": citations, "error": None}


def retrieve_docs(state: AgentState) -> AgentState:
    return _retrieve("docs", state)


def retrieve_issues(state: AgentState) -> AgentState:
    return _retrieve("issues", state)


def retrieve_platform(state: AgentState) -> AgentState:
    return _retrieve("platform", state)


# ---------------------------------------------------------------------------
# Conditional edge helpers
# ---------------------------------------------------------------------------

def route_to_partition(state: AgentState) -> str:
    """After intent classification, dispatch to the correct retrieval node."""
    return state.get("intent", "docs")  # node names match intent strings


def handle_empty_retrieval(state: AgentState) -> str:
    """After retrieval, decide whether to continue to END or loop back.

    Self-correction loop: if retrieval returned nothing AND we haven't hit the
    retry ceiling, route back to intent_router so the LLM can reconsider which
    partition to try next.
    """
    if state.get("error") and state.get("retries", 0) < MAX_RETRIES:
        logger.info(
            "[ROUTER] Empty retrieval (attempt %d/%d) — looping back to router",
            state["retries"] + 1,
            MAX_RETRIES,
        )
        # Increment retry counter before looping back
        return "retry"
    return "end"


def increment_retries(state: AgentState) -> AgentState:
    """Bump the retry counter before re-entering the router."""
    return {**state, "retries": state.get("retries", 0) + 1}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph routing graph."""
    workflow = StateGraph(AgentState)

    # ---- Nodes ----
    workflow.add_node("intent_router", intent_router)
    workflow.add_node("docs", retrieve_docs)
    workflow.add_node("issues", retrieve_issues)
    workflow.add_node("platform", retrieve_platform)
    workflow.add_node("increment_retries", increment_retries)

    # ---- Entry ----
    workflow.set_entry_point("intent_router")

    # ---- Router → partition dispatch ----
    workflow.add_conditional_edges(
        "intent_router",
        route_to_partition,
        {
            "docs": "docs",
            "issues": "issues",
            "platform": "platform",
        },
    )

    # ---- Each retrieval node → self-correction check ----
    for partition_node in ("docs", "issues", "platform"):
        workflow.add_conditional_edges(
            partition_node,
            handle_empty_retrieval,
            {
                "retry": "increment_retries",
                "end": END,
            },
        )

    # ---- Self-correction: retry counter → back to router ----
    workflow.add_edge("increment_retries", "intent_router")

    return workflow.compile()


# ---------------------------------------------------------------------------
# Quick local smoke-test  (python -m server.agent_router)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    TEST_QUESTIONS = [
        "How do I create a Kubeflow Pipeline with a GPU step?",           # → docs
        "I'm getting OOM error when running TrainingJob on 4 nodes",      # → issues
        "How do I deploy Kubeflow on OCI using Terraform modules?",       # → platform
    ]

    app = build_graph()

    for question in TEST_QUESTIONS:
        print("\n" + "=" * 70)
        print(f"Q: {question}")
        # Build a minimal initial state
        initial: AgentState = {
            "question": question,
            "intent": "",
            "context": "",
            "citations": [],
            "response": "",
            "error": None,
            "retries": 0,
        }
        result = app.invoke(initial)
        print(f"  → Intent   : {result['intent']}")
        print(f"  → Citations: {result['citations'][:2]}")
        print(f"  → Context snippet: {result['context'][:120]!r}")
        if result.get("error"):
            print(f"  ⚠ Error: {result['error']}")
