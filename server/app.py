import os
import json
import asyncio
import logging
import websockets
from websockets.server import serve
from websockets.exceptions import ConnectionClosedError
from typing import Dict, Any, List

# Config
PORT = int(os.getenv("PORT", "8000"))

# Import the agent graph
from agent import agent_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def handle_chat(message: str, websocket) -> None:
    """Handle chat using LangGraph agent with streaming"""
    try:
        print(f"[CHAT] Processing message: {message[:100]}...")
        
        # input for the graph
        inputs = {"messages": [("user", message)]}
        
        citations_collector = []
        
        # Stream events from the graph
        # We use astream_events to get granular updates (tokens, tool calls, etc.)
        async for event in agent_graph.astream_events(inputs, version="v1"):
            kind = event["event"]
            
            # Stream LLM tokens
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    await websocket.send(json.dumps({
                        "type": "content", 
                        "content": content
                    }))
            
            # Handle Tool Execution
            elif kind == "on_tool_start":
                # Maybe notify user that a tool is starting?
                # The existing API doesn't strictly require this, but we can log it.
                tool_name = event["name"]
                print(f"[TOOL] Starting: {tool_name}")
                
            elif kind == "on_tool_end":
                tool_name = event["name"]
                tool_output = str(event["data"].get("output", ""))
                
                print(f"[TOOL] Finished: {tool_name}")
                
                # Try to extract citations if it's our search tool
                # This is a bit hacky, but consistent with previous logic which parsed the output string
                # Ideally, our tool should return structured data, but LangChain tools return strings often.
                # Let's extract URLs from the output if possible using regex or simple parsing
                # For now, we just pass the tool output.
                
                await websocket.send(json.dumps({
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "content": tool_output
                }))
                
                # Attempt to extract citations from the formatted text tool output
                # The tool output format is "Source: URL\nContent: ..."
                lines = tool_output.split('\n')
                for line in lines:
                    if line.strip().startswith("Source: "):
                        url = line.strip().replace("Source: ", "")
                        if url and url not in citations_collector:
                            citations_collector.append(url)

        # Send collected citations
        if citations_collector:
            await websocket.send(json.dumps({
                "type": "citations", 
                "citations": citations_collector
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
            "content": "Connected to Kubeflow Docs Agent (LangGraph Powered)"
        }))
        
        async for message in websocket:
            try:
                # Ensure we always deal with string, not bytes
                if isinstance(message, (bytes, bytearray)):
                    message = message.decode("utf-8", errors="ignore")

                # Try to parse as JSON first
                try:
                    msg_data = json.loads(message)
                    if isinstance(msg_data, dict) and "message" in msg_data:
                        message = msg_data["message"]
                except json.JSONDecodeError:
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
    print("🚀 Starting Kubeflow Docs WebSocket Server (LangGraph)")
    print(f"   Port: {PORT}")
    
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
        
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
