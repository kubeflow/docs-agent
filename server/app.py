import os
import json
import asyncio
import httpx
import websockets
from websockets.server import serve
from websockets.exceptions import InvalidMessage, ConnectionClosedError
import logging
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# Config
KSERVE_URL = os.getenv("KSERVE_URL", "http://llama.santhosh.svc.cluster.local/openai/v1/chat/completions")
MODEL = os.getenv("MODEL", "llama3.1-8B")
PORT = int(os.getenv("PORT", "8000"))
MILVUS_HOST = "milvus-standalone-final.santhosh.svc.cluster.local"
MILVUS_PORT = "19530"
COLLECTION_NAME = "docs_rag"

# Load embedding model once
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def rag_search(query: str) -> str:
    """Search Milvus for relevant documents"""
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        query_vec = embedding_model.encode(query).tolist()
        results = collection.search(
            data=[query_vec],
            anns_field="vector", 
            param={"metric_type": "COSINE", "params": {"nprobe": 32}},
            limit=3,
            output_fields=["file_path", "content_text", "citation_url"]
        )
        
        context = ""
        for hit in results[0]:
            content = hit.entity.get('content_text', '')
            url = hit.entity.get('citation_url', '')
            context += f"Source: {url}\nContent: {content}\n\n"
        
        connections.disconnect("default")
        return context
    except Exception as e:
        return f"RAG search failed: {e}"

# System prompt to control tool usage
SYSTEM_PROMPT = """You are a helpful AI assistant with access to Kubeflow documentation through a search function.

IMPORTANT: You have two response modes:

1. KUBEFLOW MODE:
   - For questions about: Kubeflow, KServe, Katib
   - Use the search function silently (don't mention it)
   - Start responses with "[KUBEFLOW]"
   - Be detailed but concise

2. GENERAL MODE:
   - For everything else (jokes, math, chat)
   - Don't use search function
   - Start responses with "[GENERAL]"
   - Keep responses short

CRITICAL RULES:
- NEVER describe or mention the search function
- NEVER output function calls or JSON
- NEVER explain what you're going to do
- Just DO it and give the answer

Example good responses:
[KUBEFLOW] Kubeflow can be installed using...
[GENERAL] Here's a joke: Why did the...

Example BAD responses:
"I'll use the search function..."
"The function to use is..."
{any JSON or function calls}"""

RAG_TOOL = {
    "type": "function",
    "function": {
        "name": "search_kubeflow_docs",
        "description": "Search Kubeflow documentation. ONLY call this function, do not describe it. NEVER output the function call details.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
}

async def handle_chat(message: str, websocket) -> None:
    """Handle chat with RAG tool support - simple non-streaming approach"""
    
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # Single request with tools - let LLM decide if RAG is needed
            payload = {
                "model": MODEL, 
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message}
                ], 
                "tools": [RAG_TOOL],
                "tool_choice": "auto",
                "stream": False,
                "max_tokens": 1500
            }
            
            response = await client.post(KSERVE_URL, json=payload, headers={"content-type": "application/json"})
            
            if response.status_code != 200:
                await websocket.send(f"ERROR: HTTP {response.status_code}: {response.text}")
                return
            
            result = response.json()
            choice = result.get("choices", [{}])[0]
            message_obj = choice.get("message", {})
            
            # Get direct content and tool calls
            direct_content = message_obj.get("content", "")
            tool_calls = message_obj.get("tool_calls", [])

            # Handle Kubeflow questions with tool calls
            if tool_calls and any(tc["function"]["name"] == "search_kubeflow_docs" for tc in tool_calls):
                try:
                    # Get the search query
                    tool_call = next(tc for tc in tool_calls if tc["function"]["name"] == "search_kubeflow_docs")
                    args = json.loads(tool_call["function"]["arguments"])
                    query = args["query"]
                    
                    # Perform RAG search silently
                    context = rag_search(query)
                    
                    if "RAG search failed" in context:
                        # Fallback to general response
                        fallback_payload = {
                            "model": MODEL,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": message}
                            ],
                            "stream": False,
                            "max_tokens": 800
                        }
                        fallback_response = await client.post(KSERVE_URL, json=fallback_payload, headers={"content-type": "application/json"})
                        
                        if fallback_response.status_code == 200:
                            fallback_result = fallback_response.json()
                            fallback_content = fallback_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                            await websocket.send(f"[GENERAL] {fallback_content}")
                        else:
                            await websocket.send("[ERROR] Could not generate response")
                        return
                    
                    # Generate response with context
                    final_payload = {
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": message},
                            {"role": "system", "content": f"Use this documentation to answer:\n\n{context}"}
                        ],
                        "stream": False,
                        "max_tokens": 800
                    }
                    
                    final_response = await client.post(KSERVE_URL, json=final_payload, headers={"content-type": "application/json"})
                    
                    if final_response.status_code == 200:
                        final_result = final_response.json()
                        final_content = final_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        if final_content.strip():
                            # Ensure response starts with [KUBEFLOW]
                            if not final_content.startswith("[KUBEFLOW]"):
                                final_content = f"[KUBEFLOW] {final_content}"
                            await websocket.send(final_content)
                        else:
                            await websocket.send("[ERROR] Empty response from LLM")
                    else:
                        await websocket.send(f"[ERROR] Failed to generate response: {final_response.status_code}")
                        
                except Exception as e:
                    await websocket.send(f"[ERROR] Processing failed: {e}")
                    
            else:
                # Handle direct responses (non-Kubeflow)
                if direct_content.strip():
                    # Ensure response starts with [GENERAL]
                    if not direct_content.startswith("[GENERAL]"):
                        direct_content = f"[GENERAL] {direct_content}"
                    await websocket.send(direct_content)
                else:
                    await websocket.send("[ERROR] No response generated")
                    
    except Exception as e:
        await websocket.send(f"ERROR: Request failed: {e}")

async def handle_websocket(websocket, path):
    """Handle WebSocket connections with proper error handling"""
    try:
        print(f"New WebSocket connection from {websocket.remote_address}")
        async for message in websocket:
            print(f"Received message: {message[:100]}...")
            await handle_chat(message, websocket)
    except ConnectionClosedError:
        print("WebSocket connection closed normally")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send(f"ERROR: Connection error: {str(e)}")
        except:
            pass

async def health_check(path, request_headers):
    """Handle HTTP health checks"""
    if path == "/health":
        return 200, [("Content-Type", "text/plain")], b"OK"
    return None

async def main():
    print(f"Starting RAG WebSocket server on 0.0.0.0:{PORT}")
    print(f"Llama service: {KSERVE_URL}")
    print(f"Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    
    # Configure logging to reduce noise
    logging.getLogger("websockets").setLevel(logging.WARNING)
    
    async with serve(
        handle_websocket, 
        "0.0.0.0", 
        PORT,
        process_request=health_check,
        ping_interval=None,  # Disable ping/pong
        ping_timeout=None
    ):
        print("WebSocket server is running...")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())