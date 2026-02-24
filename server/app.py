import json
import asyncio
import logging
import sys
import os
import httpx

import websockets
from websockets.server import serve
from websockets.exceptions import ConnectionClosedError

# Add project root to path so shared module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.rag_core import (  # noqa: E402
    KSERVE_URL,
    MODEL,
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION,
    PORT,
    SYSTEM_PROMPT,
    TOOLS,
    execute_tool,
    deduplicate_citations,
    build_chat_payload,
)

from typing import Dict, Any, List

logger = logging.getLogger(__name__)


async def stream_llm_response(
    payload: Dict[str, Any],
    websocket,
    citations_collector: List[str] = None,
) -> None:
    """Stream response from LLM to websocket, handling tool calls"""
    if citations_collector is None:
        citations_collector = []
    try:

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", KSERVE_URL, json=payload) as response:
                if response.status_code != 200:
                    error_msg = f"LLM service error: HTTP {response.status_code}"
                    logger.error(error_msg)
                    await websocket.send(
                        json.dumps({"type": "error", "content": error_msg})
                    )
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
                            await websocket.send(
                                json.dumps(
                                    {"type": "content", "content": delta["content"]}
                                )
                            )

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

                                        # Send tool execution result to client
                                        await websocket.send(
                                            json.dumps(
                                                {
                                                    "type": "tool_result",
                                                    "tool_name": tool_call["function"][
                                                        "name"
                                                    ],
                                                    "content": result,
                                                }
                                            )
                                        )

                                        # Make follow-up request with tool results
                                        await handle_tool_follow_up(
                                            payload,
                                            tool_call,
                                            result,
                                            websocket,
                                            citations_collector,
                                        )

                                    except Exception as e:
                                        logger.error("Tool execution error: %s", e)
                                        await websocket.send(
                                            json.dumps(
                                                {
                                                    "type": "error",
                                                    "content": f"Tool execution failed: {e}",
                                                }
                                            )
                                        )

                            tool_calls_buffer.clear()
                            break  # Tool execution complete, exit streaming loop

                    except json.JSONDecodeError as e:
                        logger.error("JSON decode error: %s, line: %s", e, line)
                        continue

    except Exception as e:
        logger.error("Streaming failed: %s", e)
        await websocket.send(
            json.dumps({"type": "error", "content": f"Streaming failed: {e}"})
        )


async def handle_tool_follow_up(
    original_payload: Dict[str, Any],
    tool_call: Dict[str, Any],
    tool_result: str,
    websocket,
    citations_collector: List[str] = None,
) -> None:
    """Handle follow-up request after tool execution"""
    if citations_collector is None:
        citations_collector = []
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
        await stream_llm_response(follow_up_payload, websocket, citations_collector)

    except Exception as e:
        logger.error("Tool follow-up failed: %s", e)
        await websocket.send(
            json.dumps(
                {"type": "error", "content": f"Tool follow-up failed: {e}"}
            )
        )


async def handle_chat(message: str, websocket) -> None:
    """Handle chat with tool calling support"""
    try:
        logger.info("Processing message: %s...", message[:100])

        payload = build_chat_payload(message)

        # Collect citations throughout the conversation
        citations_collector: List[str] = []

        # Start streaming response
        await stream_llm_response(payload, websocket, citations_collector)

        # Send citations if any were collected
        if citations_collector:
            unique_citations = deduplicate_citations(citations_collector)
            await websocket.send(
                json.dumps({"type": "citations", "citations": unique_citations})
            )

        # Send completion signal
        await websocket.send(json.dumps({"type": "done"}))

    except Exception as e:
        logger.error("Chat handling failed: %s", e)
        await websocket.send(
            json.dumps({"type": "error", "content": f"Request failed: {e}"})
        )


async def handle_websocket(websocket, path):
    """Handle WebSocket connections"""
    logger.info("New connection from %s", websocket.remote_address)

    try:
        # Send welcome message
        await websocket.send(
            json.dumps(
                {
                    "type": "system",
                    "content": "Connected to Kubeflow Documentation Assistant",
                }
            )
        )

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
                    # Treat as plain text message
                    pass

                logger.info("Received: %s...", message[:100])
                await handle_chat(message, websocket)

            except Exception as e:
                logger.error("Message processing error: %s", e)
                await websocket.send(
                    json.dumps(
                        {
                            "type": "error",
                            "content": f"Message processing failed: {e}",
                        }
                    )
                )

    except ConnectionClosedError:
        logger.info("Connection closed")
    except Exception as e:
        logger.error("WebSocket error: %s", e)


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
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    # Start server
    async with serve(
        handle_websocket,
        "0.0.0.0",
        PORT,
        process_request=health_check,
        ping_interval=30,
        ping_timeout=10,
    ):
        print("✅ WebSocket server is running...")
        print(f"   WebSocket: ws://localhost:{PORT}")
        print(f"   Health: http://localhost:{PORT}/health")

        # Keep server running
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
