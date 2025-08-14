import os
import json
import asyncio
import httpx
import websockets
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

RAG_TOOL = {
    "type": "function",
    "function": {
        "name": "search_kubeflow_docs",
        "description": "Search Kubeflow documentation for specific information. Always refine the search query to be more specific.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Refined search query for Kubeflow docs"}
            },
            "required": ["query"]
        }
    }
}

async def stream_chat(message: str, websocket) -> None:
    payload = {
        "model": MODEL, 
        "messages": [{"role": "user", "content": message}], 
        "tools": [RAG_TOOL],
        "tool_choice": "auto",
        "stream": True
    }
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(KSERVE_URL, json=payload, headers={"content-type": "application/json"})
            
            if response.status_code != 200:
                await websocket.send(f"ERROR: HTTP {response.status_code}")
                await websocket.send("[DONE]")
                return
            
            tool_calls = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        choice = chunk.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        
                        # Handle tool calls
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                if tc["function"]["name"] == "search_kubeflow_docs":
                                    args = json.loads(tc["function"]["arguments"])
                                    query = args["query"]
                                    await websocket.send(f"🔍 Searching: {query}\n")
                                    
                                    context = rag_search(query)
                                    
                                    # Send final response with context
                                    final_payload = {
                                        "model": MODEL,
                                        "messages": [
                                            {"role": "user", "content": message},
                                            {"role": "assistant", "content": f"Based on the search results:\n{context}\n\nAnswer: "}
                                        ],
                                        "stream": True
                                    }
                                    
                                    final_response = await client.post(KSERVE_URL, json=final_payload, headers={"content-type": "application/json"})
                                    async for final_line in final_response.aiter_lines():
                                        if final_line.startswith("data:"):
                                            final_data = final_line[5:].strip()
                                            if final_data == "[DONE]":
                                                await websocket.send("[DONE]")
                                                return
                                            try:
                                                final_chunk = json.loads(final_data)
                                                final_content = final_chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                                                if final_content:
                                                    await websocket.send(final_content)
                                            except:
                                                pass
                                    return
                        
                        # Regular content
                        content = delta.get("content")
                        if content:
                            await websocket.send(content)
                    except:
                        pass
            
            await websocket.send("[DONE]")
    except Exception as e:
        await websocket.send(f"ERROR: {e}")
        await websocket.send("[DONE]")

async def handle_websocket(websocket):
    async for message in websocket:
        await stream_chat(message, websocket)

async def main():
    async with websockets.serve(handle_websocket, "0.0.0.0", PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())