"""
Shared fixtures for docs-agent test suite

Every mock mirrors the real objects used in server-https/app.py so that tests 
stay true to actual behaviour instead of the implementation details
"""

import os
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch


# Environment variables

@pytest.fixture(autouse=True)
def env_vars(monkeypatch):
    """Pin every environment variable the server reads at the module level

    autouse=True ensures no test accidentally reaches a real Milvus or KServe endpoint
    """
    env = {
        "KSERVE_URL": "http://test-kserve:8080/v1/chat/completions",
        "MODEL": "test-model",
        "PORT": "8000",
        "MILVUS_HOST": "localhost",
        "MILVUS_PORT": "19530",
        "MILVUS_COLLECTION": "test_collection",
        "MILVUS_VECTOR_FIELD": "vector",
        "EMBEDDING_MODEL": "sentence-transformers/all-mpnet-base-v2",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env



# SentenceTransformer encoder

@pytest.fixture()
def mock_encoder():
    """Return a fake SentenceTransformer whose .encode() produces a
    deterministic 768-dim vector, matching all-mpnet-base-v2 output shape
    """
    encoder = MagicMock()
    encoder.encode.return_value = np.random.default_rng(42).random(768).astype(np.float32)
    return encoder


@pytest.fixture()
def patch_encoder(mock_encoder):
    """Patch the SentenceTransformer constructor so the server never downloads
    real model weights during testing 
    """
    with patch(
        "server-https.app.SentenceTransformer", return_value=mock_encoder
    ) as patched:
        yield patched



# Pymilvus connections + Collection

def _make_milvus_hit(file_path, content_text, citation_url, distance=0.15):
    """Build a single Milvus hit object that behaves like the real SDK result of a search() call"""
    hit = MagicMock()
    hit.distance = distance
    hit.entity.get = lambda field, default=None: {
        "file_path": file_path,
        "content_text": content_text,
        "citation_url": citation_url,
    }.get(field, default)
    return hit


@pytest.fixture()
def sample_milvus_hits():
    """Two realistic search hits that milvus_search() would return"""
    return [
        _make_milvus_hit(
            file_path="docs/pipelines/overview.md",
            content_text="Kubeflow Pipelines is a platform for building and deploying ML workflows.",
            citation_url="https://www.kubeflow.org/docs/components/pipelines/overview/",
            distance=0.12,
        ),
        _make_milvus_hit(
            file_path="docs/pipelines/sdk.md",
            content_text="The Pipelines SDK allows you to define and manipulate pipelines programmatically.",
            citation_url="https://www.kubeflow.org/docs/components/pipelines/sdk/",
            distance=0.25,
        ),
    ]


@pytest.fixture()
def mock_collection(sample_milvus_hits):
    """A fake pymilvus.Collection pre-loaded with sample hits"""
    collection = MagicMock()
    collection.load.return_value = None
    collection.search.return_value = [sample_milvus_hits]  #list of lists of hits
    return collection


@pytest.fixture()
def patch_milvus(mock_collection):
    """Patch both pymilvus.connections and pymilvus.Collection so no real
    Milvus instance is ever contacted during testing process
    """
    with patch("server-https.app.connections") as mock_conn, \
         patch("server-https.app.Collection", return_value=mock_collection) as mock_coll_cls:
        mock_conn.connect.return_value = None
        mock_conn.disconnect.return_value = None
        yield {
            "connections": mock_conn,
            "Collection": mock_coll_cls,
            "instance": mock_collection,
        }


# httpx / KServe streaming responses

def build_sse_lines(events):
    """Convert a list of SSE event dicts into the raw lines that
    httpx.Response.aiter_lines() would yield after reading from a Kserve streaming response 
    """
    lines = []
    for event in events:
        lines.append(f"data: {json.dumps(event)}")
    lines.append("data: [DONE]")
    return lines


@pytest.fixture()
def plain_text_sse_events():
    """SSE stream for a simple text reply with no tool calls 
    matching Kserve streaming response format"""
    return [
        {"choices": [{"delta": {"content": "Kubeflow "}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Pipelines "}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "rocks."}, "finish_reason": "stop"}]},
    ]


@pytest.fixture()
def tool_call_sse_events():
    """SSE stream where the LLM decides to invoke search_kubeflow_docs"""
    return [
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "search_kubeflow_docs",
                            "arguments": "",
                        },
                    }]
                },
                "finish_reason": None,
            }]
        },
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": '{"query": "pipelines overview"}'
                        },
                    }]
                },
                "finish_reason": None,
            }]
        },
        {
            "choices": [{
                "delta": {},
                "finish_reason": "tool_calls",
            }]
        },
    ]


class FakeAsyncLineIterator:
    """An async iterator over a list of strings, mimicking
    httpx.Response.aiter_lines()
    """
    def __init__(self, lines):
        self._lines = lines
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        return line


class FakeStreamResponse:
    """Minimal stand-in for the httpx async streaming context manager"""

    def __init__(self, lines, status_code=200):
        self.status_code = status_code
        self._lines = lines

    def aiter_lines(self):
        return FakeAsyncLineIterator(self._lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeAsyncClient:
    """Replaces httpx.AsyncClient so no real HTTP calls leave from the test"""

    def __init__(self, responses=None):
        self._responses = responses or []
        self._call_index = 0

    def stream(self, method, url, **kwargs):
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture()
def fake_kserve_plain(plain_text_sse_events):
    """An httpx.AsyncClient that returns a plain-text SSE stream"""
    lines = build_sse_lines(plain_text_sse_events)
    return FakeAsyncClient(responses=[FakeStreamResponse(lines)])


@pytest.fixture()
def fake_kserve_tool(tool_call_sse_events, plain_text_sse_events):
    """An httpx.AsyncClient that first returns a tool call stream, then a
    plaintext follow-up stream, simulating the two request tool call process 
    """
    tool_lines = build_sse_lines(tool_call_sse_events)
    followup_lines = build_sse_lines(plain_text_sse_events)
    return FakeAsyncClient(responses=[
        FakeStreamResponse(tool_lines),
        FakeStreamResponse(followup_lines),
    ])


@pytest.fixture()
def fake_kserve_error():
    """An httpx.AsyncClient whose response returns HTTP 500 error"""
    return FakeAsyncClient(responses=[FakeStreamResponse([], status_code=500)])


# Sample payloads

@pytest.fixture()
def sample_tool_call():
    """A fully assembled tool_call dict as stream_llm_response would build it
    after buffering streaming chunks from kserve
    """
    return {
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "search_kubeflow_docs",
            "arguments": json.dumps({"query": "pipelines overview", "top_k": 5}),
        },
    }


@pytest.fixture()
def unknown_tool_call():
    """A tool_call referencing a function that doesn't exist in the registry of the server"""
    return {
        "id": "call_unknown_999",
        "type": "function",
        "function": {
            "name": "nonexistent_tool",
            "arguments": json.dumps({"foo": "bar"}),
        },
    }


@pytest.fixture()
def chat_payload():
    """The JSON payload that the /chat endpoint builds before calling the LLM"""
    from importlib import import_module
    # Avoid importing at top level so env_vars fixture can take effect first
    mod = import_module("server-https.app")
    return {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": mod.SYSTEM_PROMPT},
            {"role": "user", "content": "How do I install Kubeflow Pipelines?"},
        ],
        "tools": mod.TOOLS,
        "tool_choice": "auto",
        "stream": True,
        "max_tokens": 1500,
    }


# FastAPI test client
@pytest.fixture()
def client():
    """HTTPX AsyncClient wired to the FastAPI app for integration testing"""
    from httpx import ASGITransport, AsyncClient
    from importlib import import_module

    mod = import_module("server-https.app")
    transport = ASGITransport(app=mod.app)
    return AsyncClient(transport=transport, base_url="http://testserver")
