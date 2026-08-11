#!/usr/bin/env python3
"""End-to-end MCP smoke: initialize session and call all three search tools."""

from __future__ import annotations

import json
import os
import re
import sys

import requests

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")
MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}

TOOL_CALLS = [
    ("search_kubeflow_docs", {"query": "what is kubeflow pipelines", "top_k": 1}),
    ("search_github_issues", {"query": "installation error", "top_k": 1}),
    ("search_kubeflow_code", {"query": "InferenceService deployment", "top_k": 1}),
]


def _parse_sse_json(text: str) -> dict:
    """Extract the last JSON-RPC result object from an SSE response body."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                return json.loads(payload)
    return json.loads(text)


def mcp_session() -> dict[str, str]:
    init = requests.post(
        MCP_URL,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "mcp-smoke", "version": "1.0"},
            },
        },
        headers=MCP_HEADERS,
        timeout=60,
    )
    init.raise_for_status()
    session = init.headers.get("mcp-session-id") or init.headers.get("Mcp-Session-Id")
    if not session:
        raise RuntimeError("MCP initialize succeeded but no mcp-session-id header")
    headers = {**MCP_HEADERS, "Mcp-Session-Id": session}
    requests.post(
        MCP_URL,
        json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        headers=headers,
        timeout=60,
    ).raise_for_status()
    return headers


def tools_list(headers: dict[str, str]) -> None:
    resp = requests.post(
        MCP_URL,
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.text
    for name in ("search_kubeflow_docs", "search_github_issues", "search_kubeflow_code"):
        if name not in body:
            raise RuntimeError(f"tools/list missing {name}")


def tools_call(headers: dict[str, str], name: str, arguments: dict) -> str:
    resp = requests.post(
        MCP_URL,
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        headers=headers,
        timeout=180,
    )
    resp.raise_for_status()
    data = _parse_sse_json(resp.text)
    if "error" in data:
        raise RuntimeError(f"tools/call {name} error: {data['error']}")
    result = data.get("result", {})
    content = result.get("content", [])
    if not content:
        raise RuntimeError(f"tools/call {name} returned no content")
    text_parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    text = "\n".join(text_parts).strip()
    if not text:
        raise RuntimeError(f"tools/call {name} returned empty text")
    if text.startswith("Search failed:"):
        raise RuntimeError(text)
    return text


def main() -> int:
    headers = mcp_session()
    tools_list(headers)
    print("mcp tools/list ok")

    for tool_name, args in TOOL_CALLS:
        text = tools_call(headers, tool_name, args)
        preview = re.sub(r"\s+", " ", text)[:120]
        print(f"mcp tools/call {tool_name} ok — {preview!r}")

    emb_url = os.environ.get("EMBEDDINGS_URL", "").strip()
    if emb_url:
        emb = requests.post(
            emb_url,
            json={"inputs": ["kubeflow pipelines smoke test"]},
            timeout=120,
        )
        emb.raise_for_status()
        vecs = emb.json()
        if not isinstance(vecs, list) or len(vecs) != 1 or len(vecs[0]) != 768:
            raise RuntimeError("unexpected embeddings response: " + repr(vecs)[:200])
        print("embeddings smoke ok", emb_url, "dim", len(vecs[0]))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"mcp smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
