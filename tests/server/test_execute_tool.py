#Tests for execute_tool() in server-https/app.py

import json
import pytest
from importlib import import_module
from unittest.mock import patch


@pytest.fixture(autouse=True)
def _load_app(env_vars):
    pass


def _get_execute_tool():
    mod = import_module("server-https.app")
    return mod.execute_tool


@pytest.mark.asyncio
class TestExecuteToolSearchKubeflowDocs:

    async def test_returns_formatted_text_and_citations(self, patch_encoder, patch_milvus, sample_tool_call):
        result_text, citations = await _get_execute_tool()(sample_tool_call)

        assert "docs/pipelines/overview.md" in result_text
        assert "docs/pipelines/sdk.md" in result_text
        assert "https://www.kubeflow.org/docs/components/pipelines/overview/" in citations
        assert "https://www.kubeflow.org/docs/components/pipelines/sdk/" in citations

    async def test_formatted_text_contains_similarity_scores(self, patch_encoder, patch_milvus, sample_tool_call):
        result_text, _ = await _get_execute_tool()(sample_tool_call)

        # similarity = 1 - distance; first hit distance=0.12 -> 0.880
        assert "0.880" in result_text

    async def test_no_duplicate_citations(self, patch_encoder, patch_milvus):
        """If multiple hits share the same citation_url, it appears only once."""
        from tests.conftest import _make_milvus_hit

        same_url = "https://www.kubeflow.org/docs/components/pipelines/overview/"
        hits = [
            _make_milvus_hit("a.md", "text a", same_url, 0.1),
            _make_milvus_hit("b.md", "text b", same_url, 0.2),
        ]
        patch_milvus["instance"].search.return_value = [hits]

        tool_call = {
            "id": "call_dup",
            "type": "function",
            "function": {
                "name": "search_kubeflow_docs",
                "arguments": json.dumps({"query": "pipelines"}),
            },
        }
        _, citations = await _get_execute_tool()(tool_call)
        assert citations.count(same_url) == 1

    async def test_empty_results_returns_no_relevant(self, patch_encoder, patch_milvus):
        patch_milvus["instance"].search.return_value = [[]]

        tool_call = {
            "id": "call_empty",
            "type": "function",
            "function": {
                "name": "search_kubeflow_docs",
                "arguments": json.dumps({"query": "nonexistent topic"}),
            },
        }
        result_text, citations = await _get_execute_tool()(tool_call)
        assert result_text == "No relevant results found."
        assert citations == []


@pytest.mark.asyncio
class TestExecuteToolUnknown:

    async def test_unknown_tool_returns_error_string(self, unknown_tool_call):
        result_text, citations = await _get_execute_tool()(unknown_tool_call)
        assert result_text == "Unknown tool: nonexistent_tool"
        assert citations == []


@pytest.mark.asyncio
class TestExecuteToolMalformedInput:

    async def test_bad_json_arguments_returns_error(self):
        tool_call = {
            "id": "call_bad",
            "type": "function",
            "function": {
                "name": "search_kubeflow_docs",
                "arguments": "not valid json{{{",
            },
        }
        result_text, citations = await _get_execute_tool()(tool_call)
        assert "Tool execution failed" in result_text
        assert citations == []
