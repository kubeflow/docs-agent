from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import json

# ── Shared setup ─────────────────────────────────────────
connections.connect("default", host="localhost", port="19530")
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ── Core search function ──────────────────────────────────
def search_index(collection_name: str, question: str, top_k: int = 3) -> list:
    vector     = embed_model.encode(question).tolist()
    collection = Collection(collection_name)
    collection.load()
    results    = collection.search(
        data=[vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["content_text", "source_url", "h1", "h2"]
    )
    chunks = []
    for hit in results[0]:
        chunks.append({
            "text":       hit.entity.get("content_text", ""),
            "source_url": hit.entity.get("source_url", ""),
            "h1":         hit.entity.get("h1", ""),
            "h2":         hit.entity.get("h2", ""),
            "score":      round(hit.score, 4)
        })
    return chunks


# ── MCP Tool 1: query_docs ───────────────────────────────
def query_docs_tool(question: str, top_k: int = 3) -> dict:
    """
    MCP tool: search Kubeflow documentation index.

    Use for: conceptual questions, tutorials, architecture explanations.
    Returns: golden snippet (150 tokens) + validation URL per result.

    This implements the Thin Context pattern from gsoc2026_agentic_rag.md:
    - Returns minimal context to preserve caller's token budget
    - Includes direct source URL for caller to fetch full context locally
    """
    chunks = search_index("docs_index", question, top_k)

    results = []
    for chunk in chunks:
        # Thin Context: truncate to ~150 tokens (600 chars)
        golden_snippet   = chunk["text"][:600]
        validation_link  = chunk["source_url"]

        results.append({
            "golden_snippet":   golden_snippet,
            "validation_link":  validation_link,
            "section":          f"{chunk['h1']} > {chunk['h2']}",
            "relevance_score":  chunk["score"]
        })

    return {
        "tool":     "query_docs",
        "question": question,
        "results":  results,
        "count":    len(results)
    }


# ── MCP Tool 2: query_code ───────────────────────────────
def query_code_tool(question: str, top_k: int = 3) -> dict:
    """
    MCP tool: search Kubeflow manifests code index.

    Use for: YAML configs, Kubernetes resources, debugging, API questions.
    Returns: golden snippet (150 tokens) + direct GitHub file URL.

    The validation_link points directly to the file in kubeflow/manifests
    so the caller's IDE agent can fetch the full file for broader context.
    """
    chunks = search_index("code_index", question, top_k)

    results = []
    for chunk in chunks:
        golden_snippet  = chunk["text"][:600]
        validation_link = chunk["source_url"]

        results.append({
            "golden_snippet":   golden_snippet,
            "validation_link":  validation_link,
            "kind":             chunk["h1"],
            "name":             chunk["h2"],
            "relevance_score":  chunk["score"]
        })

    return {
        "tool":     "query_code",
        "question": question,
        "results":  results,
        "count":    len(results)
    }


# ── MCP Tool 3: query_both ───────────────────────────────
def query_both_tool(question: str, top_k: int = 2) -> dict:
    """
    MCP tool: search both docs and code indexes simultaneously.

    Use for: complex questions needing both conceptual and technical context.
    """
    doc_results  = query_docs_tool(question, top_k)
    code_results = query_code_tool(question, top_k)

    return {
        "tool":         "query_both",
        "question":     question,
        "docs_results": doc_results["results"],
        "code_results": code_results["results"],
        "total_count":  doc_results["count"] + code_results["count"]
    }


# ── MCP server ───────────────────────────────────────────
def create_mcp_server():
    """
    Creates a FastMCP server exposing all three tools
    as standard MCP endpoints callable from any IDE.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("kubeflow-docs-agent")

    @mcp.tool()
    def query_docs(question: str, top_k: int = 3) -> str:
        """Search Kubeflow documentation for conceptual questions."""
        result = query_docs_tool(question, top_k)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def query_code(question: str, top_k: int = 3) -> str:
        """Search Kubeflow manifests for YAML configs and code."""
        result = query_code_tool(question, top_k)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def query_both(question: str, top_k: int = 2) -> str:
        """Search both docs and code indexes simultaneously."""
        result = query_both_tool(question, top_k)
        return json.dumps(result, indent=2)

    return mcp