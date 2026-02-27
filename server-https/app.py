import json
import logging
import sys
import os
from typing import Dict, Any, List, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import shared module, supporting both installed-package and repo layouts
try:
    from shared.rag_core import (
        KSERVE_URL,
        MODEL,
        MILVUS_HOST,
        MILVUS_PORT,
        MILVUS_COLLECTION,
        PORT,
        execute_tool,
        deduplicate_citations,
        build_chat_payload,
    )
except ModuleNotFoundError as _original_exc:
    # Fallback for development: add project root so ../shared is importable
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    try:
        from shared.rag_core import (
            KSERVE_URL,
            MODEL,
            MILVUS_HOST,
            MILVUS_PORT,
            MILVUS_COLLECTION,
            PORT,
            execute_tool,
            deduplicate_citations,
            build_chat_payload,
        )
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Could not import 'shared.rag_core'. Ensure the 'shared/' directory "
            "is available on PYTHONPATH or copied into the runtime environment "
            "(for Docker, include 'COPY shared/ /app/shared/')."
        ) from _original_exc

logger = logging.getLogger(__name__)

app = FastAPI(title="Kubeflow Docs API Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API key authentication middleware
# ---------------------------------------------------------------------------
# Set the API_KEY env var to enable auth.  When unset, all requests are
# allowed through (backwards-compatible with the current deployment).
# Health and OPTIONS endpoints are always unauthenticated.

_API_KEY: str = os.getenv("API_KEY", "")

_PUBLIC_PATHS = frozenset({"/health", "/", "/openapi.json", "/docs", "/redoc"})


@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    """Reject requests that lack a valid API key (when API_KEY is set)."""
    if not _API_KEY:
        # Auth disabled — pass through
        return await call_next(request)

    # Always allow health checks, OPTIONS preflight, and docs
    if request.url.path in _PUBLIC_PATHS or request.method == "OPTIONS":
        return await call_next(request)

    # Accept key from Authorization header or X-API-Key header
    auth_header = request.headers.get("Authorization", "")
    x_api_key = request.headers.get("X-API-Key", "")

    token = ""
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    elif x_api_key:
        token = x_api_key

    if token != _API_KEY:
        logger.warning(
            "Rejected unauthenticated request to %s from %s",
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key."},
        )

    return await call_next(request)


class ChatRequest(BaseModel):
    message: str
    stream: bool = True


async def stream_llm_response(
    payload: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """Stream response from LLM and handle tool calls, yielding SSE events"""
    citations_collector: List[str] = []

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", KSERVE_URL, json=payload) as response:
                if response.status_code != 200:
                    error_msg = f"LLM service error: HTTP {response.status_code}"
                    logger.error(error_msg)
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
                                            "name": tool_call.get("function", {}).get(
                                                "name", ""
                                            ),
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
                                    tool_calls_buffer[index]["function"]["name"] = (
                                        function_data["name"]
                                    )
                                if "arguments" in function_data:
                                    tool_calls_buffer[index]["function"][
                                        "arguments"
                                    ] += function_data["arguments"]

                        # Handle regular content
                        elif "content" in delta and delta["content"]:
                            yield f"data: {json.dumps({'type': 'content', 'content': delta['content']})}\n\n"

                        # Handle finish reason - execute tools if needed
                        if finish_reason == "tool_calls":
                            logger.info(
                                "Finish reason: tool_calls, executing %d tools",
                                len(tool_calls_buffer),
                            )

                            # Execute all accumulated tool calls
                            for tool_call in tool_calls_buffer.values():
                                if (
                                    tool_call["function"]["name"]
                                    and tool_call["function"]["arguments"]
                                ):
                                    try:
                                        logger.info(
                                            "Executing: %s",
                                            tool_call["function"]["name"],
                                        )

                                        result, tool_citations = await execute_tool(
                                            tool_call
                                        )

                                        # Collect citations
                                        citations_collector.extend(tool_citations)

                                        # Send tool execution result
                                        yield f"data: {json.dumps({'type': 'tool_result', 'tool_name': tool_call['function']['name'], 'content': result})}\n\n"

                                        # Make follow-up request with tool results
                                        async for follow_up_chunk in handle_tool_follow_up(
                                            payload,
                                            tool_call,
                                            result,
                                            citations_collector,
                                        ):
                                            yield follow_up_chunk

                                    except Exception as e:
                                        logger.error("Tool execution error: %s", e)
                                        yield f"data: {json.dumps({'type': 'error', 'content': f'Tool execution failed: {e}'})}\n\n"

                            tool_calls_buffer.clear()
                            break  # Tool execution complete, exit streaming loop

                    except json.JSONDecodeError as e:
                        logger.error("JSON decode error: %s, line: %s", e, line)
                        continue

        # Send citations if any were collected
        if citations_collector:
            unique_citations = deduplicate_citations(citations_collector)
            yield f"data: {json.dumps({'type': 'citations', 'citations': unique_citations})}\n\n"

        # Send completion signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error("Streaming failed: %s", e)
        yield f"data: {json.dumps({'type': 'error', 'content': f'Streaming failed: {e}'})}\n\n"


async def handle_tool_follow_up(
    original_payload: Dict[str, Any],
    tool_call: Dict[str, Any],
    tool_result: str,
    citations_collector: List[str],
) -> AsyncGenerator[str, None]:
    """Handle follow-up request after tool execution"""
    try:
        logger.info("Handling follow-up request with tool results")

        # Create messages with tool call and result
        messages = original_payload["messages"].copy()

        # Add assistant's tool call message
        messages.append({"role": "assistant", "tool_calls": [tool_call]})

        # Add tool result message
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result,
            }
        )

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
        logger.error("Tool follow-up failed: %s", e)
        yield f"data: {json.dumps({'type': 'error', 'content': f'Tool follow-up failed: {e}'})}\n\n"


async def get_non_streaming_response(
    payload: Dict[str, Any],
) -> tuple[str, List[str]]:
    """Get non-streaming response by collecting all streaming chunks"""
    response_content = ""
    citations: List[str] = []

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
                        status_code=500,
                        detail=data.get("content", "Unknown error"),
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
        logger.info("Processing message: %s...", request.message[:100])

        payload = build_chat_payload(request.message)

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
            unique_citations = deduplicate_citations(citations)

            return {
                "response": response_content,
                "citations": unique_citations if unique_citations else None,
            }

    except Exception as e:
        logger.error("Chat handling failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Request failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting Kubeflow Docs HTTP API Server")
    print(f"   Port: {PORT}")
    print(f"   LLM Service: {KSERVE_URL}")
    print(f"   Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"   Collection: {MILVUS_COLLECTION}")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
