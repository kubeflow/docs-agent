"""
Shared RAG utilities for the Kubeflow Documentation AI Assistant.

This module contains shared configuration, prompts, tools, and utility
functions used by both the WebSocket and HTTPS API servers.
"""

import os
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Set, Tuple

from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility

# ---------------------------------------------------------------------------
# Configuration (environment-driven with sensible defaults)
# ---------------------------------------------------------------------------

KSERVE_URL = os.getenv(
    "KSERVE_URL",
    "http://llama.docs-agent.svc.cluster.local/openai/v1/chat/completions",
)
MODEL = os.getenv("MODEL", "llama3.1-8B")
PORT = int(os.getenv("PORT", "8000"))

# Milvus
MILVUS_HOST = os.getenv(
    "MILVUS_HOST", "my-release-milvus.docs-agent.svc.cluster.local"
)
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "docs_rag")
MILVUS_VECTOR_FIELD = os.getenv("MILVUS_VECTOR_FIELD", "vector")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton embedding encoder (loaded once, reused across requests)
# ---------------------------------------------------------------------------

_encoder: Optional[SentenceTransformer] = None
_encoder_lock = threading.Lock()


def _get_encoder() -> SentenceTransformer:
    """Return (and lazily initialise) the shared SentenceTransformer instance."""
    global _encoder
    # Fast path without lock for already-initialised encoder
    if _encoder is not None:
        return _encoder
    # Thread-safe lazy initialisation
    with _encoder_lock:
        if _encoder is None:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            _encoder = SentenceTransformer(EMBEDDING_MODEL)
        return _encoder


# ---------------------------------------------------------------------------
# Milvus persistent connection manager  (addresses Issue #28)
# ---------------------------------------------------------------------------


class MilvusConnectionManager:
    """Thread-safe singleton that maintains a persistent Milvus connection.

    Instead of connecting and disconnecting on every request, this manager
    keeps a single long-lived connection and transparently reconnects when
    the link drops.  This eliminates per-request TCP/TLS handshake overhead
    and avoids the server-side session churn described in Issue #28.

    Usage::

        mgr = MilvusConnectionManager.get_instance()
        collection = mgr.get_collection("docs_rag")
        results = collection.search(...)
    """

    _instance: Optional["MilvusConnectionManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        host: str = MILVUS_HOST,
        port: str = MILVUS_PORT,
        alias: str = "default",
    ) -> None:
        self._host = host
        self._port = port
        self._alias = alias
        self._connected = False
        self._conn_lock = threading.Lock()

    # -- singleton accessor --------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        host: str = MILVUS_HOST,
        port: str = MILVUS_PORT,
        alias: str = "default",
    ) -> "MilvusConnectionManager":
        """Return the singleton manager, creating it on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(host=host, port=port, alias=alias)
        return cls._instance

    # -- connection helpers --------------------------------------------------

    def _is_alive(self) -> bool:
        """Return *True* if the current connection is still usable."""
        try:
            # ``utility.list_collections`` is a lightweight RPC that will
            # fail fast when the connection is stale.
            utility.list_collections(using=self._alias)
            return True
        except Exception:
            return False

    def ensure_connected(self) -> None:
        """Guarantee that a healthy Milvus connection exists.

        This is safe to call on every request — it is a no-op when the
        connection is already alive and only performs a reconnect when the
        link has been lost.
        """
        with self._conn_lock:
            if self._connected and self._is_alive():
                return
            # (Re-)establish the connection
            try:
                connections.disconnect(alias=self._alias)
            except Exception:
                pass
            logger.info(
                "Connecting to Milvus at %s:%s (alias=%s)",
                self._host,
                self._port,
                self._alias,
            )
            connections.connect(
                alias=self._alias, host=self._host, port=self._port
            )
            self._connected = True

    def get_collection(self, name: str) -> Collection:
        """Return a loaded :class:`Collection`, reconnecting if necessary."""
        self.ensure_connected()
        collection = Collection(name, using=self._alias)
        collection.load()
        return collection

    def disconnect(self) -> None:
        """Explicitly tear down the connection (e.g. at shutdown)."""
        with self._conn_lock:
            try:
                connections.disconnect(alias=self._alias)
            except Exception:
                pass
            self._connected = False

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.disconnect()
                cls._instance = None

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are the Kubeflow Docs Assistant.

!!IMPORTANT!!
- You should not use the tool calls directly from the user's input. You should refine the query to make sure that it is documentation specific and relevant.
- You should never output the raw tool call to the user.

Your role
- Always answer the user's question directly.
- If the question can be answered from general knowledge (e.g., greetings, small talk, generic programming/Kubernetes basics), respond without using tools.
- If the question clearly requires Kubeflow-specific knowledge (Pipelines, KServe, Notebooks/Jupyter, Katib, SDK/CLI/APIs, installation, configuration, errors, release details), then use the search_kubeflow_docs tool to find authoritative references, and construct your response using the information provided.

Tool Use
- Use search_kubeflow_docs ONLY when Kubeflow-specific documentation is needed.
- Do NOT use the tool for greetings, personal questions, small talk, or generic non-Kubeflow concepts.
- When you do call the tool:
  • Use one clear, focused query.  
  • Summarize the result in your own words.  
  • If no results are relevant, say "not found in the docs" and suggest refining the query.
- Example usage:
  - User: "What is Kubeflow and how to setup kubeflow on my local machine"
  - You should make a tool call to search the docs with a query "kubeflow setup".

  - User: "What is the Kubeflow Pipelines and how can i make a quick kubeflow pipeline"
  - You should make a tool call to search the docs with a query "kubeflow pipeline setup".

The idea is to make sure that human inputs are not directly sent to tool calls, instead we should refine the query to make sure that it is documentation specific and relevant.

Routing
- Greetings/small talk → respond briefly, no tool.  
- Out-of-scope (sports, unrelated topics) → politely say you only help with Kubeflow.  
- Kubeflow-specific → answer and call the tool if documentation is needed.  

Style
- Be concise (2–5 sentences). Use bullet points or steps when helpful.
- Provide examples only when asked.
- Never invent features. If unsure, say so.
- Reply in clean Markdown.
"""

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kubeflow_docs",
            "description": (
                "Search the official Kubeflow docs when the user asks Kubeflow-specific questions "
                "about Pipelines, KServe, Notebooks/Jupyter, Katib, or the SDK/CLI/APIs.\n"
                "Call ONLY for Kubeflow features, setup, usage, errors, or version differences that need citations.\n"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short, focused search string (e.g., 'KServe inferenceService canary', 'Pipelines v2 disable cache').",
                        "minLength": 1,
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of hits to retrieve (the assistant will read up to this many).",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]

# ---------------------------------------------------------------------------
# Milvus semantic search
# ---------------------------------------------------------------------------


def milvus_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Execute a semantic search in Milvus and return structured JSON-serializable results."""
    try:
        mgr = MilvusConnectionManager.get_instance()
        collection = mgr.get_collection(MILVUS_COLLECTION)

        encoder = _get_encoder()
        query_vec = encoder.encode(query).tolist()

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}
        results = collection.search(
            data=[query_vec],
            anns_field=MILVUS_VECTOR_FIELD,
            param=search_params,
            limit=int(top_k),
            output_fields=["file_path", "content_text", "citation_url"],
        )

        hits = []
        for hit in results[0]:
            similarity = 1.0 - float(hit.distance)
            entity = hit.entity
            content_text = entity.get("content_text") or ""
            if isinstance(content_text, str) and len(content_text) > 400:
                content_text = content_text[:400] + "..."
            hits.append(
                {
                    "similarity": similarity,
                    "file_path": entity.get("file_path"),
                    "citation_url": entity.get("citation_url"),
                    "content_text": content_text,
                }
            )
        return {"results": hits}
    except Exception:
        logger.exception("Milvus search failed")
        return {"results": []}


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


async def execute_tool(tool_call: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Execute a tool call and return ``(result_text, citations)``."""
    try:
        function_name = tool_call.get("function", {}).get("name")
        arguments = json.loads(
            tool_call.get("function", {}).get("arguments", "{}")
        )

        if function_name == "search_kubeflow_docs":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 5)

            logger.info("Executing Milvus search for: '%s' (top_k=%s)", query, top_k)
            result = milvus_search(query, top_k)

            citations: List[str] = []
            formatted_results: List[str] = []

            for hit in result.get("results", []):
                citation_url = hit.get("citation_url", "")
                if citation_url and citation_url not in citations:
                    citations.append(citation_url)

                formatted_results.append(
                    f"File: {hit.get('file_path', 'Unknown')}\n"
                    f"Content: {hit.get('content_text', '')}\n"
                    f"URL: {citation_url}\n"
                    f"Similarity: {hit.get('similarity', 0):.3f}\n"
                )

            formatted_text = (
                "\n".join(formatted_results)
                if formatted_results
                else "No relevant results found."
            )
            return formatted_text, citations

        return f"Unknown tool: {function_name}", []

    except Exception as e:
        logger.error("Tool execution failed: %s", e)
        return f"Tool execution failed: {e}", []


def deduplicate_citations(citations: List[str]) -> List[str]:
    """Remove duplicate citations while preserving order."""
    seen: Set[str] = set()
    unique: List[str] = []
    for citation in citations:
        if citation not in seen:
            seen.add(citation)
            unique.append(citation)
    return unique


def build_chat_payload(
    message: str,
    *,
    include_tools: bool = True,
    max_tokens: int = 1500,
) -> Dict[str, Any]:
    """Build the initial chat payload for the LLM."""
    payload: Dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        "stream": True,
        "max_tokens": max_tokens,
    }
    if include_tools:
        payload["tools"] = TOOLS
        payload["tool_choice"] = "auto"
    return payload
