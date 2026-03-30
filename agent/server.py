import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from agent.core.graph import agent as langgraph_agent
from agent.tools.mcp_tools import query_docs_tool, query_code_tool, query_both_tool

# ── App setup ─────────────────────────────────────────────
app = FastAPI(
    title="Kubeflow Docs Agent",
    description="Agentic RAG for Kubeflow documentation and code",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request model ─────────────────────────────────────────
class ChatRequest(BaseModel):
    message:  str
    stream:   Optional[bool] = True
    backend:  Optional[str]  = "langgraph"  # langgraph or mcp

# ── Health check ──────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

# ── MCP tool endpoints (callable from any IDE) ────────────
@app.post("/mcp/query_docs")
async def mcp_query_docs(request: Request):
    body     = await request.json()
    question = body.get("question", "")
    top_k    = body.get("top_k", 3)
    result   = query_docs_tool(question, top_k)
    return result

@app.post("/mcp/query_code")
async def mcp_query_code(request: Request):
    body     = await request.json()
    question = body.get("question", "")
    top_k    = body.get("top_k", 3)
    result   = query_code_tool(question, top_k)
    return result

@app.post("/mcp/query_both")
async def mcp_query_both(request: Request):
    body     = await request.json()
    question = body.get("question", "")
    top_k    = body.get("top_k", 2)
    result   = query_both_tool(question, top_k)
    return result

# ── Main chat endpoint ────────────────────────────────────
@app.post("/chat")
async def chat(request: ChatRequest):
    if request.stream:
        return StreamingResponse(
            stream_response(request.message),
            media_type="text/event-stream"
        )
    else:
        result = langgraph_agent.invoke({
            "question":  request.message,
            "route":     "",
            "chunks":    [],
            "citations": [],
            "answer":    "",
            "messages":  []
        })
        return {
            "answer":    result["answer"],
            "route":     result["route"],
            "citations": result["citations"],
            "chunks":    len(result["chunks"])
        }

# ── SSE streaming response ────────────────────────────────
async def stream_response(question: str):
    # Send thinking update
    yield f"data: {json.dumps({'type': 'status', 'content': 'Analyzing question...'})}\n\n"
    await asyncio.sleep(0.1)

    # Run agent
    result = langgraph_agent.invoke({
        "question":  question,
        "route":     "",
        "chunks":    [],
        "citations": [],
        "answer":    "",
        "messages":  []
    })

    # Send route decision
    route_msg = f"Searching {result['route']} index..."
    yield f"data: {json.dumps({'type': 'status', 'content': route_msg})}\n\n"
    await asyncio.sleep(0.1)

    # Stream answer word by word
    words = result["answer"].split()
    for i, word in enumerate(words):
        chunk = word + " "
        yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
        if i % 10 == 0:
            await asyncio.sleep(0.01)

    # Send citations
    yield f"data: {json.dumps({'type': 'citations', 'citations': result['citations']})}\n\n"

    # Send done
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)