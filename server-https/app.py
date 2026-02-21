import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
from pydantic import BaseModel
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
import uvicorn

# Config
KSERVE_URL = os.getenv(
    "KSERVE_URL", "http://llama.santhosh.svc.cluster.local/openai/v1/chat/completions"
)
MODEL = os.getenv("MODEL", "llama3.1-8B")
PORT = int(os.getenv("PORT", "8000"))

# Milvus Config
MILVUS_HOST = os.getenv("MILVUS_HOST", "my-release-milvus.docs-agent.svc.cluster.local")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "docs_rag")
MILVUS_VECTOR_FIELD = os.getenv("MILVUS_VECTOR_FIELD", "vector")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# System prompt (same as WebSocket version)
SYSTEM_PROMPT = """
You are the Kubeflow Docs Assistant.

!!IMPORTANT!!
- You should not use the tool calls directly from the user's input. You should refine the query to make sure that it is documenation specific and relevant.
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

The idea is to make sure that human inputs are not directly sent to tool calls, instead we should refine the query to make sure that it is documenation specific and relevant.

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

app = FastAPI(title="Kubeflow Docs API Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    stream: Optional[bool] = True


def milvus_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Execute a semantic search in Milvus and return structured JSON serializable results."""
    try:
        # Connect to Milvus
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(MILVUS_COLLECTION)
        collection.load()

        # Encoder (same model as pipeline)
        encoder = SentenceTransformer(EMBEDDING_MODEL)
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
            # similarity = 1 - distance for COSINE in Milvus
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
    except Exception as e:
        print(f"[ERROR] Milvus search failed: {e}")
        return {"results": []}
    finally:
        try:
            connections.disconnect(alias="default")
        except Exception:
            pass


async def execute_tool(tool_call: Dict[str, Any]) -> tuple[str, List[str]]:
    """Execute a tool call and return the result and citations"""
    try:
        function_name = tool_call.get("function", {}).get("name")
        arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))

        if function_name == "search_kubeflow_docs":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 5)

            print(f"[TOOL] Executing Milvus search for: '{query}' (top_k={top_k})")
            result = milvus_search(query, top_k)
            
            # Collect citations
            citations = []
            formatted_results = []

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
                "\n".join(formatted_results) if formatted_results else "No relevant results found."
            )
            return formatted_text, citations

        return f"Unknown tool: {function_name}", []

    except Exception as e:
        print(f"[ERROR] Tool execution failed: {e}")
        return f"Tool execution failed: {e}", []


async def stream_llm_response(payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """Stream response from LLM and handle tool calls, yielding SSE events"""
    citations_collector = []

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", KSERVE_URL, json=payload) as response:
                if response.status_code != 200:
                    error_msg = f"LLM service error: HTTP {response.status_code}"
                    print(f"[ERROR] {error_msg}")
                    yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                    return

                # Buffer for accumulating tool calls
                tool_calls_buffer = {}

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        finish_reason = choices[0].get("finish_reason")

                        # Handle tool calls in streaming
                        if "tool_calls" in delta:
                            tool_calls = delta["tool_calls"]
                            for tool_call in tool_calls:
                                index = tool_call.get("index", 0)

                                # Initialize tool call buffer if needed
                                if index not in tool_calls_buffer:
                                    tool_calls_buffer[index] = {
                                        "id": tool_call.get("id", ""),
                                        "type": tool_call.get("type", "function"),
                                        "function": {
                                            "name": tool_call.get("function", {}).get("name", ""),
                                            "arguments": "",
                                        },
                                    }

                                # Update tool call data
                                if tool_call.get("id"):
                                    tool_calls_buffer[index]["id"] = tool_call["id"]
                                if tool_call.get("type"):
                                    tool_calls_buffer[index]["type"] = tool_call["type"]

                                function_data = tool_call.get("function", {})
                                if function_data.get("name"):
                                    tool_calls_buffer[index]["function"]["name"] = function_data[
                                        "name"
                                    ]
                                if "arguments" in function_data:
                                    tool_calls_buffer[index]["function"]["arguments"] += (
                                        function_data["arguments"]
                                    )

                        # Handle regular content
                        elif "content" in delta and delta["content"]:
                            yield f"data: {json.dumps({'type': 'content', 'content': delta['content']})}\n\n"

                        # Handle finish reason - execute tools if needed
                        if finish_reason == "tool_calls":
                            print(
                                f"[TOOL] Finish reason: tool_calls, executing {len(tool_calls_buffer)} tools"
                            )

                            # Execute all accumulated tool calls
                            for tool_call in tool_calls_buffer.values():
                                if (
                                    tool_call["function"]["name"]
                                    and tool_call["function"]["arguments"]
                                ):
                                    try:
                                        print(f"[TOOL] Executing: {tool_call['function']['name']}")
                                        print(
                                            f"[TOOL] Arguments: {tool_call['function']['arguments']}"
                                        )

                                        result, tool_citations = await execute_tool(tool_call)

                                        # Collect citations
                                        citations_collector.extend(tool_citations)

                                        # Send tool execution result
                                        yield f"data: {json.dumps({'type': 'tool_result', 'tool_name': tool_call['function']['name'], 'content': result})}\n\n"

                                        # Make follow-up request with tool results
                                        async for follow_up_chunk in handle_tool_follow_up(
                                            payload, tool_call, result, citations_collector
                                        ):
                                            yield follow_up_chunk

                                    except Exception as e:
                                        print(f"[ERROR] Tool execution error: {e}")
                                        yield f"data: {json.dumps({'type': 'error', 'content': f'Tool execution failed: {e}'})}\n\n"

                            tool_calls_buffer.clear()
                            break  # Tool execution complete, exit streaming loop

                    except json.JSONDecodeError as e:
                        print(f"[ERROR] JSON decode error: {e}, line: {line}")
                        continue

        # Send citations if any were collected
        if citations_collector:
            # Remove duplicates while preserving order
            unique_citations = []
            for citation in citations_collector:
                if citation not in unique_citations:
                    unique_citations.append(citation)

            yield f"data: {json.dumps({'type': 'citations', 'citations': unique_citations})}\n\n"

        # Send completion signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        print(f"[ERROR] Streaming failed: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'Streaming failed: {e}'})}\n\n"


async def handle_tool_follow_up(
    original_payload: Dict[str, Any],
    tool_call: Dict[str, Any],
    tool_result: str,
    citations_collector: List[str],
) -> AsyncGenerator[str, None]:
    """Handle follow-up request after tool execution"""
    try:
        print("[TOOL] Handling follow-up request with tool results")

        # Create messages with tool call and result
        messages = original_payload["messages"].copy()

        # Add assistant's tool call message
        messages.append({"role": "assistant", "tool_calls": [tool_call]})

        # Add tool result message
        messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": tool_result})

        # Create follow-up payload - remove tools to get final response
        follow_up_payload = {
            "model": original_payload["model"],
            "messages": messages,
            "stream": True,
            "max_tokens": 1000,
        }

        # Stream the follow-up response
        async for chunk in stream_llm_response(follow_up_payload):
            yield chunk

    except Exception as e:
        print(f"[ERROR] Tool follow-up failed: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'Tool follow-up failed: {e}'})}\n\n"


async def get_non_streaming_response(payload: Dict[str, Any]) -> tuple[str, List[str]]:
    """Get non-streaming response by collecting all streaming chunks"""
    response_content = ""
    citations = []

    async for chunk in stream_llm_response(payload):
        if chunk.startswith("data: "):
            try:
                data = json.loads(chunk[6:].strip())
                if data.get("type") == "content":
                    response_content += data.get("content", "")
                elif data.get("type") == "citations":
                    citations.extend(data.get("citations", []))
                elif data.get("type") == "error":
                    raise HTTPException(
                        status_code=500, detail=data.get("content", "Unknown error")
                    )
            except json.JSONDecodeError:
                continue

    return response_content, citations


@app.get("/")
async def hello():
    """Simple hello endpoint"""
    return {"message": "Hello from Kubeflow Docs API!", "service": "https-api"}


@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes probes"""
    return {"status": "healthy", "service": "https-api"}


@app.options("/chat")
async def options_chat():
    """Handle preflight OPTIONS request"""
    return {"message": "OK"}


@app.options("/")
async def options_root():
    """Handle preflight OPTIONS request for root"""
    return {"message": "OK"}


@app.options("/health")
async def options_health():
    """Handle preflight OPTIONS request for health"""
    return {"message": "OK"}


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with RAG capabilities - supports both streaming and non-streaming"""
    try:
        print(f"[CHAT] Processing message: {request.message[:100]}...")

        # Create initial payload
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message},
            ],
            "tools": TOOLS,
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 1500,
        }

        if request.stream:
            # Return streaming response using Server-Sent Events
            return StreamingResponse(
                stream_llm_response(payload),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control",
                },
            )
        else:
            # Return non-streaming JSON response
            response_content, citations = await get_non_streaming_response(payload)

            # Remove duplicates from citations while preserving order
            unique_citations = []
            for citation in citations:
                if citation not in unique_citations:
                    unique_citations.append(citation)

            return {
                "response": response_content,
                "citations": unique_citations if unique_citations else None,
            }

    except Exception as e:
        print(f"[ERROR] Chat handling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Request failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting Kubeflow Docs HTTP API Server")
    print(f"   Port: {PORT}")
    print(f"   LLM Service: {KSERVE_URL}")
    print(f"   Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"   Collection: {MILVUS_COLLECTION}")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
