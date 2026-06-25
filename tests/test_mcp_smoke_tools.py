"""Tests for MCP smoke helper (SSE parsing)."""

import json
import sys
from pathlib import Path

MCP_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "mcp-server"
sys.path.insert(0, str(MCP_DIR))

import smoke_tools  # noqa: E402


def test_parse_sse_json_from_data_lines():
    body = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"ok"}]}}\n\n'
    data = smoke_tools._parse_sse_json(body)
    assert data["result"]["content"][0]["text"] == "ok"


def test_parse_plain_json_body():
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
    data = smoke_tools._parse_sse_json(body)
    assert data["result"]["ok"] is True
