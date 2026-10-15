"""Tests for MCP tracing (docs-agent-mcp/mcp-server/flo_tracing.py). No Langfuse needed."""

import asyncio
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("pymilvus", MagicMock())
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs-agent-mcp" / "mcp-server"))

import flo_tracing  # noqa: E402


@pytest.fixture
def recorder(monkeypatch):
    calls = []

    class Obs:
        def update(self, output):
            calls[-1]["output"] = output

    @contextmanager
    def fake_observe(as_type, name, input, **extra):
        calls.append({"type": as_type, "name": name, "input": input, "extra": extra})
        yield Obs()

    monkeypatch.setattr(flo_tracing, "_observe", fake_observe)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    return calls


def test_disabled_without_keys_is_a_plain_call(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    @flo_tracing.traced("tool", inputs=("q",))
    def f(q, n=1):
        return q * n

    assert f("ab", n=2) == "abab"


def test_records_selected_inputs_and_output(recorder):
    @flo_tracing.traced("retriever", inputs=("query", "top_k"), output=lambda hits: {"n": len(hits)})
    def search(collection, query, top_k=5, output_fields=None):
        return [{"distance": 0.9}, {"distance": 0.5}]

    assert len(search("docs", "katib", top_k=3, output_fields=["a"])) == 2
    assert recorder == [
        {
            "type": "retriever",
            "name": "search",
            "input": {"query": "katib", "top_k": 3},
            "extra": {},
            "output": {"n": 2},
        }
    ]


def test_output_errors_never_break_the_call(recorder):
    @flo_tracing.traced("tool", inputs=(), output=lambda r: 1 / 0)
    def f():
        return "ok"

    assert f() == "ok"


def test_retriever_output_summary():
    hits = [{"id": 1, "distance": 0.8, "entity": {"content_text": "x"}}, {"entity": {"score": 0.4}}, {"id": 3}]
    out = flo_tracing.retriever_output(hits, mode="hybrid", intent="howto")
    assert out == {"hit_count": 3, "top_scores": [0.8, 0.4], "mode": "hybrid", "intent": "howto"}


def test_mcp_tool_schemas_keep_their_parameters():
    import server

    async def tools():
        return {t.name: t for t in await server.mcp.list_tools()}

    listed = asyncio.run(tools())
    assert set(listed) == {"search_kubeflow_docs", "search_github_issues", "search_kubeflow_code"}
    params = listed["search_kubeflow_code"].parameters["properties"]
    assert {"query", "top_k", "resource_kind", "repo"} <= set(params)
