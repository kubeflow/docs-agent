import os
import json
import asyncio
import httpx
import websockets
from websockets.server import serve
from websockets.exceptions import ConnectionClosedError
import logging
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

# Config
KSERVE_URL = os.getenv("KSERVE_URL", "http://llama.santhosh.svc.cluster.local/openai/v1/chat/completions")
MODEL = os.getenv("MODEL", "llama3.1-8B")
PORT = int(os.getenv("PORT", "8000"))

# Milvus Config
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-standalone-final.santhosh.svc.cluster.local")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "docs_rag")
MILVUS_VECTOR_FIELD = os.getenv("MILVUS_VECTOR_FIELD", "vector")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# System prompt
SYSTEM_PROMPT = """
You are the Kubeflow Docs Assistant.

Your role
- Always answer the user's question directly.
- If the question can be answered from general knowledge (e.g., greetings, small talk, generic programming/Kubernetes basics), respond without using tools.
- If the question clearly requires Kubeflow-specific knowledge (Pipelines, KServe, Notebooks/Jupyter, Katib, SDK/CLI/APIs, installation, configuration, errors, release details), then use the search_kubeflow_docs tool to find authoritative references, and include them in your answer.

Tool Use
- Use search_kubeflow_docs ONLY when Kubeflow-specific documentation is needed.
- Do NOT use the tool for greetings, personal questions, small talk, or generic non-Kubeflow concepts.
- When you do call the tool:
  • Use one clear, focused query.  
  • Summarize the result in your own words.  
  • Include 1–2 citations from the returned URLs.  
  • If no results are relevant, say “not found in the docs” and suggest refining the query.

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
            hits.append({
                "similarity": similarity,
                "file_path": entity.get("file_path"),
                "citation_url": entity.get("citation_url"),
                "content_text": content_text,
            })
        return {"results": hits}
    except Exception as e:
        print(f"[ERROR] Milvus search failed: {e}")
        return {"results": []}
    finally:
        try:
            connections.disconnect(alias="default")
        except Exception:
            pass
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
                        "minLength": 1
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of hits to retrieve (the assistant will read up to this many).",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    }
]


async def execute_tool(tool_call: Dict[str, Any]) -> tuple[str, List[str]]:
    """Execute a tool call and return the result and citations"""
    try:
        function_name = tool_call.get("function", {}).get("name")
        arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
        
        if function_name == "search_kubeflow_docs":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 5)
            
            print(f"[TOOL] Executing Milvus search for: '{query}' (top_k={top_k})")
            result = milvus_search(query, 15)
            
            # Collect citations
            citations = []
            formatted_results = []
            
            for hit in result.get("results", []):
                citation_url = hit.get('citation_url', '')
                if citation_url and citation_url not in citations:
                    citations.append(citation_url)
                
                formatted_results.append(
                    f"File: {hit.get('file_path', 'Unknown')}\n"
                    f"Content: {hit.get('content_text', '')}\n"
                    f"URL: {citation_url}\n"
                    f"Similarity: {hit.get('similarity', 0):.3f}\n"
                )
            
            formatted_text = "\n".join(formatted_results) if formatted_results else "No relevant results found."
            return formatted_text, citations
        
        return f"Unknown tool: {function_name}", []
        
    except Exception as e:
        print(f"[ERROR] Tool execution failed: {e}")
        return f"Tool execution failed: {e}", []

async def stream_llm_response(payload: Dict[str, Any], websocket, citations_collector: List[str] = None) -> None:
    """Stream response from LLM to websocket, handling tool calls"""
    if citations_collector is None:
        citations_collector = []
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", KSERVE_URL, json=payload) as response:
                if response.status_code != 200:
                    error_msg = f"LLM service error: HTTP {response.status_code}"
                    print(f"[ERROR] {error_msg}")
                    await websocket.send(json.dumps({"type": "error", "content": error_msg}))
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
                                            "arguments": ""
                                        }
                                    }
                                
                                # Update tool call data
                                if tool_call.get("id"):
                                    tool_calls_buffer[index]["id"] = tool_call["id"]
                                if tool_call.get("type"):
                                    tool_calls_buffer[index]["type"] = tool_call["type"]
                                
                                function_data = tool_call.get("function", {})
                                if function_data.get("name"):
                                    tool_calls_buffer[index]["function"]["name"] = function_data["name"]
                                if "arguments" in function_data:
                                    tool_calls_buffer[index]["function"]["arguments"] += function_data["arguments"]
                        
                        # Handle regular content
                        elif "content" in delta and delta["content"]:
                            await websocket.send(json.dumps({
                                "type": "content", 
                                "content": delta["content"]
                            }))
                        
                        # Handle finish reason - execute tools if needed
                        if finish_reason == "tool_calls":
                            print(f"[TOOL] Finish reason: tool_calls, executing {len(tool_calls_buffer)} tools")
                            
                            # Execute all accumulated tool calls
                            for tool_call in tool_calls_buffer.values():
                                if tool_call["function"]["name"] and tool_call["function"]["arguments"]:
                                    try:
                                        print(f"[TOOL] Executing: {tool_call['function']['name']}")
                                        print(f"[TOOL] Arguments: {tool_call['function']['arguments']}")
                                        
                                        result, tool_citations = await execute_tool(tool_call)
                                        
                                        # Collect citations
                                        citations_collector.extend(tool_citations)
                                        
                                        # Send tool execution result to client
                                        await websocket.send(json.dumps({
                                            "type": "tool_result",
                                            "tool_name": tool_call["function"]["name"],
                                            "content": result
                                        }))
                                        
                                        # Make follow-up request with tool results
                                        await handle_tool_follow_up(payload, tool_call, result, websocket, citations_collector)
                                        
                                    except Exception as e:
                                        print(f"[ERROR] Tool execution error: {e}")
                                        await websocket.send(json.dumps({
                                            "type": "error",
                                            "content": f"Tool execution failed: {e}"
                                        }))
                            
                            tool_calls_buffer.clear()
                            break  # Tool execution complete, exit streaming loop
                            
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] JSON decode error: {e}, line: {line}")
                        continue
                        
    except Exception as e:
        print(f"[ERROR] Streaming failed: {e}")
        await websocket.send(json.dumps({"type": "error", "content": f"Streaming failed: {e}"}))

async def handle_tool_follow_up(original_payload: Dict[str, Any], tool_call: Dict[str, Any], tool_result: str, websocket, citations_collector: List[str] = None) -> None:
    """Handle follow-up request after tool execution"""
    if citations_collector is None:
        citations_collector = []
    try:
        print("[TOOL] Handling follow-up request with tool results")
        
        # Create messages with tool call and result
        messages = original_payload["messages"].copy()
        
        # Add assistant's tool call message
        messages.append({
            "role": "assistant",
            "tool_calls": [tool_call]
        })
        
        # Add tool result message
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": tool_result
        })
        
        # Create follow-up payload - remove tools to get final response
        follow_up_payload = {
            "model": original_payload["model"],
            "messages": messages,
            "stream": True,
            "max_tokens": 1000
        }
        
        # Stream the follow-up response
        await stream_llm_response(follow_up_payload, websocket, citations_collector)
        
    except Exception as e:
        print(f"[ERROR] Tool follow-up failed: {e}")
        await websocket.send(json.dumps({"type": "error", "content": f"Tool follow-up failed: {e}"}))

async def handle_chat(message: str, websocket) -> None:
    """Handle chat with tool calling support"""
    try:
        print(f"[CHAT] Processing message: {message[:100]}...")
        
        # Create initial payload
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
            "tools": TOOLS,
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 1500
        }
        
        # Collect citations throughout the conversation
        citations_collector = []
        
        # Start streaming response
        await stream_llm_response(payload, websocket, citations_collector)
        
        # Send citations if any were collected
        if citations_collector:
            # Remove duplicates while preserving order
            unique_citations = []
            for citation in citations_collector:
                if citation not in unique_citations:
                    unique_citations.append(citation)
            
            await websocket.send(json.dumps({
                "type": "citations", 
                "citations": unique_citations
            }))
        
        # Send completion signal
        await websocket.send(json.dumps({"type": "done"}))
        
    except Exception as e:
        print(f"[ERROR] Chat handling failed: {e}")
        await websocket.send(json.dumps({"type": "error", "content": f"Request failed: {e}"}))

async def handle_websocket(websocket, path):
    """Handle WebSocket connections"""
    print(f"[WS] New connection from {websocket.remote_address}")
    
    try:
        # Send welcome message
        await websocket.send(json.dumps({
            "type": "system",
            "content": "Connected to Kubeflow Documentation Assistant"
        }))
        
        async for message in websocket:
            try:
                # Try to parse as JSON first
                try:
                    msg_data = json.loads(message)
                    if isinstance(msg_data, dict) and "message" in msg_data:
                        message = msg_data["message"]
                except json.JSONDecodeError:
                    # Treat as plain text message
                    pass
                
                print(f"[WS] Received: {message[:100]}...")
                await handle_chat(message, websocket)
                
            except Exception as e:
                print(f"[ERROR] Message processing error: {e}")
                await websocket.send(json.dumps({
                    "type": "error", 
                    "content": f"Message processing failed: {e}"
                }))
                
    except ConnectionClosedError:
        print("[WS] Connection closed")
    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")

async def health_check(path, request_headers):
    """Handle HTTP health checks"""
    if path == "/health":
        return 200, [("Content-Type", "text/plain")], b"OK"
    return None

async def main():
    """Start the WebSocket server"""
    print("🚀 Starting Kubeflow Docs WebSocket Server")
    print(f"   Port: {PORT}")
    print(f"   LLM Service: {KSERVE_URL}")
    print(f"   Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"   Collection: {MILVUS_COLLECTION}")
    
    # Configure logging
    logging.getLogger("websockets").setLevel(logging.WARNING)
    
    # Start server
    async with serve(
        handle_websocket, 
        "0.0.0.0", 
        PORT,
        process_request=health_check,
        ping_interval=30,
        ping_timeout=10
    ):
        print("✅ WebSocket server is running...")
        print(f"   WebSocket: ws://localhost:{PORT}")
        print(f"   Health: http://localhost:{PORT}/health")
        
        # Keep server running
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
