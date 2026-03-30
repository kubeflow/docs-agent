from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import operator
import json
import datetime

# ── Connect to Milvus ────────────────────────────────────
connections.connect("default", host="localhost", port="19530")
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ── State definition ─────────────────────────────────────
class AgentState(TypedDict):
    question:  str
    route:     str
    chunks:    list
    citations: list
    answer:    str
    messages:  Annotated[list, operator.add]

# ── Helper: search Milvus ────────────────────────────────
def search_index(collection_name: str, question: str, top_k: int = 3) -> list:
    vector = embed_model.encode(question).tolist()
    collection = Collection(collection_name)
    collection.load()
    results = collection.search(
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

# ── Node 1: Router ───────────────────────────────────────
def router(state: AgentState) -> dict:
    q = state["question"].lower()

    code_keywords = [
        "yaml", "manifest", "crd", "deployment", "service",
        "bug", "error", "crash", "fix", "issue", "exception",
        "code", "function", "class", "api", "webhook", "config"
    ]
    doc_keywords = [
        "what is", "how does", "explain", "overview", "concept",
        "architecture", "introduction", "guide", "tutorial"
    ]

    code_score = sum(1 for w in code_keywords if w in q)
    doc_score  = sum(1 for w in doc_keywords  if w in q)

    if code_score > doc_score:
        route = "code"
    elif doc_score > code_score:
        route = "docs"
    else:
        route = "both"

    # Emit routing log — future training data
    log = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "question":  state["question"],
        "route":     route,
        "doc_score": doc_score,
        "code_score": code_score
    }
    with open("routing_logs.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")

    print(f"[Router] route={route} doc_score={doc_score} code_score={code_score}")
    return {"route": route}

# ── Routing function ─────────────────────────────────────
def decide_tool(state: AgentState) -> Literal["query_docs", "query_code", "query_both"]:
    return {
        "docs": "query_docs",
        "code": "query_code",
        "both": "query_both"
    }[state["route"]]

# ── Node 2a: Query docs index ────────────────────────────
def query_docs(state: AgentState) -> dict:
    print("[Tool] searching docs_index...")
    chunks    = search_index("docs_index", state["question"])
    citations = list(set(c["source_url"] for c in chunks if c["source_url"]))
    return {"chunks": chunks, "citations": citations}

# ── Node 2b: Query code index ────────────────────────────
def query_code(state: AgentState) -> dict:
    print("[Tool] searching code_index...")
    chunks    = search_index("code_index", state["question"])
    citations = list(set(c["source_url"] for c in chunks if c["source_url"]))
    return {"chunks": chunks, "citations": citations}

# ── Node 2c: Query both indexes ──────────────────────────
def query_both(state: AgentState) -> dict:
    print("[Tool] searching both indexes...")
    doc_chunks  = search_index("docs_index", state["question"], top_k=2)
    code_chunks = search_index("code_index", state["question"], top_k=2)
    chunks      = doc_chunks + code_chunks
    citations   = list(set(c["source_url"] for c in chunks if c["source_url"]))
    return {"chunks": chunks, "citations": citations}

# ── Node 3: Synthesize ───────────────────────────────────
def synthesize(state: AgentState) -> dict:
    context = "\n\n".join([
        f"Source: {c['source_url']}\nSection: {c['h1']} > {c['h2']}\n{c['text']}"
        for c in state["chunks"]
    ])

    # Stub — Phase 2 Part 3 replaces this with real LLM call
    answer = (
        f"Based on {len(state['chunks'])} retrieved chunks "
        f"from the {state['route']} index:\n\n"
        f"{context[:800]}\n\n"
        f"Citations: {', '.join(state['citations'])}"
    )
    return {"answer": answer}

# ── Build the graph ──────────────────────────────────────
def build_agent():
    builder = StateGraph(AgentState)

    builder.add_node("router",     router)
    builder.add_node("query_docs", query_docs)
    builder.add_node("query_code", query_code)
    builder.add_node("query_both", query_both)
    builder.add_node("synthesize", synthesize)

    builder.set_entry_point("router")

    builder.add_conditional_edges(
        "router",
        decide_tool,
        {
            "query_docs": "query_docs",
            "query_code": "query_code",
            "query_both": "query_both",
        }
    )

    builder.add_edge("query_docs", "synthesize")
    builder.add_edge("query_code", "synthesize")
    builder.add_edge("query_both", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()

agent = build_agent()