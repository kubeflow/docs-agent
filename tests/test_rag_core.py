"""Tests for shared.rag_core — the core RAG utility module.

These tests validate the public API of ``shared.rag_core`` without
requiring a live Milvus instance or GPU.  Heavy dependencies are mocked
in ``conftest.py`` before this module is imported.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

# conftest.py installs mocks before this import resolves
from shared import rag_core
from shared.rag_core import (
    MilvusConnectionManager,
    _get_encoder,
    build_chat_payload,
    deduplicate_citations,
    execute_tool,
    milvus_search,
    SYSTEM_PROMPT,
    TOOLS,
    MODEL,
)


# ====================================================================
# build_chat_payload
# ====================================================================


class TestBuildChatPayload:
    """Tests for :func:`build_chat_payload`."""

    def test_returns_dict_with_required_keys(self):
        payload = build_chat_payload("hello")
        assert isinstance(payload, dict)
        assert "model" in payload
        assert "messages" in payload
        assert "stream" in payload

    def test_model_from_env(self):
        payload = build_chat_payload("hello")
        assert payload["model"] == MODEL

    def test_messages_structure(self):
        payload = build_chat_payload("What is KServe?")
        messages = payload["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is KServe?"

    def test_tools_included_by_default(self):
        payload = build_chat_payload("hello")
        assert "tools" in payload
        assert payload["tools"] == TOOLS
        assert payload["tool_choice"] == "auto"

    def test_tools_excluded_when_disabled(self):
        payload = build_chat_payload("hello", include_tools=False)
        assert "tools" not in payload
        assert "tool_choice" not in payload

    def test_custom_max_tokens(self):
        payload = build_chat_payload("hello", max_tokens=500)
        assert payload["max_tokens"] == 500

    def test_default_max_tokens(self):
        payload = build_chat_payload("hello")
        assert payload["max_tokens"] == 1500

    def test_streaming_enabled(self):
        payload = build_chat_payload("hello")
        assert payload["stream"] is True


# ====================================================================
# deduplicate_citations
# ====================================================================


class TestDeduplicateCitations:
    """Tests for :func:`deduplicate_citations`."""

    def test_empty_list(self):
        assert deduplicate_citations([]) == []

    def test_no_duplicates(self):
        urls = ["https://a.com", "https://b.com"]
        assert deduplicate_citations(urls) == urls

    def test_removes_duplicates(self):
        urls = [
            "https://a.com",
            "https://b.com",
            "https://a.com",
            "https://c.com",
            "https://b.com",
        ]
        assert deduplicate_citations(urls) == [
            "https://a.com",
            "https://b.com",
            "https://c.com",
        ]

    def test_preserves_order(self):
        urls = ["https://c.com", "https://a.com", "https://b.com", "https://a.com"]
        result = deduplicate_citations(urls)
        assert result == ["https://c.com", "https://a.com", "https://b.com"]

    def test_single_item(self):
        assert deduplicate_citations(["https://a.com"]) == ["https://a.com"]

    def test_all_duplicates(self):
        urls = ["https://a.com"] * 5
        assert deduplicate_citations(urls) == ["https://a.com"]


# ====================================================================
# _get_encoder (singleton)
# ====================================================================


class TestGetEncoder:
    """Tests for the lazy-singleton :func:`_get_encoder`."""

    def test_returns_encoder_instance(self):
        enc = _get_encoder()
        assert enc is not None
        # Should have an encode() method
        assert hasattr(enc, "encode")

    def test_singleton_returns_same_instance(self):
        enc1 = _get_encoder()
        enc2 = _get_encoder()
        assert enc1 is enc2

    def test_encoder_produces_vector(self):
        enc = _get_encoder()
        vec = enc.encode("test query")
        assert isinstance(vec, list)
        assert len(vec) == 768


# ====================================================================
# MilvusConnectionManager
# ====================================================================


class TestMilvusConnectionManager:
    """Tests for :class:`MilvusConnectionManager`."""

    def test_singleton_pattern(self):
        mgr1 = MilvusConnectionManager.get_instance()
        mgr2 = MilvusConnectionManager.get_instance()
        assert mgr1 is mgr2

    def test_reset_instance(self):
        mgr1 = MilvusConnectionManager.get_instance()
        MilvusConnectionManager.reset_instance()
        mgr2 = MilvusConnectionManager.get_instance()
        assert mgr1 is not mgr2

    def test_ensure_connected_calls_connect(self):
        mgr = MilvusConnectionManager.get_instance()
        # _is_alive will return False (MagicMock default is truthy, but we
        # start with _connected=False, so the first branch triggers connect)
        assert mgr._connected is False
        mgr.ensure_connected()
        assert mgr._connected is True

    def test_disconnect_resets_state(self):
        mgr = MilvusConnectionManager.get_instance()
        mgr.ensure_connected()
        assert mgr._connected is True
        mgr.disconnect()
        assert mgr._connected is False

    def test_get_collection_ensures_connection(self):
        mgr = MilvusConnectionManager.get_instance()
        assert mgr._connected is False
        # get_collection internally calls ensure_connected
        mgr.get_collection("test_collection")
        assert mgr._connected is True

    def test_custom_host_port(self):
        MilvusConnectionManager.reset_instance()
        mgr = MilvusConnectionManager.get_instance(
            host="custom-host", port="12345", alias="custom"
        )
        assert mgr._host == "custom-host"
        assert mgr._port == "12345"
        assert mgr._alias == "custom"


# ====================================================================
# execute_tool
# ====================================================================


class TestExecuteTool:
    """Tests for :func:`execute_tool`."""

    def _make_tool_call(self, name: str, arguments: dict) -> dict:
        return {
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            }
        }

    def test_unknown_tool(self):
        tool_call = self._make_tool_call("nonexistent_tool", {})
        result_text, citations = asyncio.get_event_loop().run_until_complete(
            execute_tool(tool_call)
        )
        assert "Unknown tool" in result_text
        assert citations == []

    @patch.object(rag_core, "milvus_search")
    def test_search_kubeflow_docs_returns_formatted_results(self, mock_search):
        mock_search.return_value = {
            "results": [
                {
                    "file_path": "docs/pipelines.md",
                    "content_text": "Kubeflow Pipelines is a platform...",
                    "citation_url": "https://kubeflow.org/docs/pipelines",
                    "similarity": 0.95,
                },
            ]
        }
        tool_call = self._make_tool_call(
            "search_kubeflow_docs", {"query": "kubeflow pipelines"}
        )
        result_text, citations = asyncio.get_event_loop().run_until_complete(
            execute_tool(tool_call)
        )
        assert "docs/pipelines.md" in result_text
        assert "https://kubeflow.org/docs/pipelines" in citations

    @patch.object(rag_core, "milvus_search")
    def test_search_with_no_results(self, mock_search):
        mock_search.return_value = {"results": []}
        tool_call = self._make_tool_call(
            "search_kubeflow_docs", {"query": "nonexistent topic"}
        )
        result_text, citations = asyncio.get_event_loop().run_until_complete(
            execute_tool(tool_call)
        )
        assert "No relevant results found" in result_text
        assert citations == []

    @patch.object(rag_core, "milvus_search")
    def test_search_respects_top_k(self, mock_search):
        mock_search.return_value = {"results": []}
        tool_call = self._make_tool_call(
            "search_kubeflow_docs", {"query": "kserve", "top_k": 3}
        )
        asyncio.get_event_loop().run_until_complete(execute_tool(tool_call))
        mock_search.assert_called_once_with("kserve", 3)

    def test_malformed_tool_call_returns_error(self):
        # Missing 'function' key entirely
        tool_call = {}
        result_text, citations = asyncio.get_event_loop().run_until_complete(
            execute_tool(tool_call)
        )
        # Should not crash — returns an error message
        assert isinstance(result_text, str)

    @patch.object(rag_core, "milvus_search")
    def test_multiple_citations_collected(self, mock_search):
        mock_search.return_value = {
            "results": [
                {
                    "file_path": "a.md",
                    "content_text": "content a",
                    "citation_url": "https://a.com",
                    "similarity": 0.9,
                },
                {
                    "file_path": "b.md",
                    "content_text": "content b",
                    "citation_url": "https://b.com",
                    "similarity": 0.8,
                },
                {
                    "file_path": "c.md",
                    "content_text": "content c",
                    "citation_url": "https://a.com",  # duplicate
                    "similarity": 0.7,
                },
            ]
        }
        tool_call = self._make_tool_call(
            "search_kubeflow_docs", {"query": "test"}
        )
        _, citations = asyncio.get_event_loop().run_until_complete(
            execute_tool(tool_call)
        )
        # execute_tool does its own dedup internally
        assert citations == ["https://a.com", "https://b.com"]


# ====================================================================
# milvus_search (mocked Milvus)
# ====================================================================


class TestMilvusSearch:
    """Tests for :func:`milvus_search` with mocked Milvus backend."""

    def _make_hit(self, distance, file_path, content_text, citation_url):
        hit = MagicMock()
        hit.distance = distance
        hit.entity.get = lambda key, default=None: {
            "file_path": file_path,
            "content_text": content_text,
            "citation_url": citation_url,
        }.get(key, default)
        return hit

    @patch.object(rag_core.MilvusConnectionManager, "get_instance")
    def test_returns_results_dict(self, mock_get_instance):
        mock_mgr = MagicMock()
        mock_get_instance.return_value = mock_mgr

        mock_collection = MagicMock()
        mock_mgr.get_collection.return_value = mock_collection

        hit = self._make_hit(0.05, "docs/test.md", "Test content", "https://test.com")
        mock_collection.search.return_value = [[hit]]

        result = milvus_search("test query", top_k=1)
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["file_path"] == "docs/test.md"
        assert result["results"][0]["citation_url"] == "https://test.com"

    @patch.object(rag_core.MilvusConnectionManager, "get_instance")
    def test_similarity_calculation(self, mock_get_instance):
        mock_mgr = MagicMock()
        mock_get_instance.return_value = mock_mgr
        mock_collection = MagicMock()
        mock_mgr.get_collection.return_value = mock_collection

        # distance=0.1 should yield similarity=0.9
        hit = self._make_hit(0.1, "f.md", "text", "https://url.com")
        mock_collection.search.return_value = [[hit]]

        result = milvus_search("query")
        assert abs(result["results"][0]["similarity"] - 0.9) < 1e-6

    @patch.object(rag_core.MilvusConnectionManager, "get_instance")
    def test_content_truncation(self, mock_get_instance):
        mock_mgr = MagicMock()
        mock_get_instance.return_value = mock_mgr
        mock_collection = MagicMock()
        mock_mgr.get_collection.return_value = mock_collection

        long_content = "x" * 500
        hit = self._make_hit(0.0, "f.md", long_content, "https://url.com")
        mock_collection.search.return_value = [[hit]]

        result = milvus_search("query")
        content = result["results"][0]["content_text"]
        # Should be truncated to 400 chars + "..."
        assert len(content) == 403
        assert content.endswith("...")

    @patch.object(rag_core.MilvusConnectionManager, "get_instance")
    def test_search_failure_returns_empty(self, mock_get_instance):
        mock_get_instance.side_effect = Exception("Connection refused")
        result = milvus_search("query")
        assert result == {"results": []}

    @patch.object(rag_core.MilvusConnectionManager, "get_instance")
    def test_top_k_forwarded(self, mock_get_instance):
        mock_mgr = MagicMock()
        mock_get_instance.return_value = mock_mgr
        mock_collection = MagicMock()
        mock_mgr.get_collection.return_value = mock_collection
        mock_collection.search.return_value = [[]]

        milvus_search("query", top_k=7)
        call_kwargs = mock_collection.search.call_args
        assert call_kwargs.kwargs.get("limit") == 7 or call_kwargs[1].get("limit") == 7
