"""Tests for the MCP server (kagent-feast-mcp/mcp-server/server.py).

Mocking strategy: We pre-populate sys.modules with mock versions of
sentence_transformers and pymilvus BEFORE importing server.py. This avoids
loading the real heavy libraries (which take 30s+ and need GPU/model downloads).
The mocks persist for the lifetime of the test process — this is intentional
since only the server module references them.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add the MCP server directory to the path
MCP_SERVER_DIR = Path(__file__).parent.parent / "kagent-feast-mcp" / "mcp-server"
sys.path.insert(0, str(MCP_SERVER_DIR))

# Pre-mock heavy dependencies before importing server.
# server.py does `from pymilvus import MilvusClient` and
# `from sentence_transformers import SentenceTransformer` at import time.
# These mocks are never cleaned up — server retains the mock bindings
# for the test process lifetime, which is exactly what we want.
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pymilvus", MagicMock())

if "server" in sys.modules:
    del sys.modules["server"]
import server


@pytest.fixture(autouse=True)
def reset_server_globals():
    """Reset server globals before each test so state doesn't leak."""
    original_model = server.model
    original_client = server.client
    original_st = server.SentenceTransformer
    original_mc = server.MilvusClient
    yield
    server.model = original_model
    server.client = original_client
    server.SentenceTransformer = original_st
    server.MilvusClient = original_mc


@pytest.fixture
def inject_mocks(mock_milvus_client, mock_sentence_transformer):
    """Inject mock model and client into the server module."""
    server.model = mock_sentence_transformer
    server.client = mock_milvus_client
    return mock_milvus_client, mock_sentence_transformer


class TestInit:
    """Tests for the _init() lazy initialization function."""

    def test_init_creates_model_and_client_when_none(self):
        """_init() should create model and client when they are None."""
        server.model = None
        server.client = None

        mock_st_class = MagicMock(return_value=MagicMock())
        mock_mc_class = MagicMock(return_value=MagicMock())
        server.SentenceTransformer = mock_st_class
        server.MilvusClient = mock_mc_class

        server._init()

        mock_st_class.assert_called_once_with(server.EMBEDDING_MODEL)
        mock_mc_class.assert_called_once_with(
            uri=server.MILVUS_URI,
            user=server.MILVUS_USER,
            password=server.MILVUS_PASSWORD,
        )

    def test_init_is_idempotent(self):
        """Calling _init() twice should not create duplicate clients."""
        server.model = None
        server.client = None

        mock_st_class = MagicMock(return_value=MagicMock())
        mock_mc_class = MagicMock(return_value=MagicMock())
        server.SentenceTransformer = mock_st_class
        server.MilvusClient = mock_mc_class

        server._init()
        server._init()

        mock_st_class.assert_called_once()
        mock_mc_class.assert_called_once()

    def test_init_skips_if_already_initialized(self):
        """_init() should not overwrite existing model/client."""
        existing_model = MagicMock()
        existing_client = MagicMock()
        server.model = existing_model
        server.client = existing_client

        mock_st_class = MagicMock()
        mock_mc_class = MagicMock()
        server.SentenceTransformer = mock_st_class
        server.MilvusClient = mock_mc_class

        server._init()

        assert server.model is existing_model
        assert server.client is existing_client
        mock_st_class.assert_not_called()
        mock_mc_class.assert_not_called()


class TestSearchKubeflowDocs:
    """Tests for the search_kubeflow_docs MCP tool."""

    def test_returns_no_results_message_when_empty(self, inject_mocks):
        """Should return 'No results found' when Milvus returns empty."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        result = server.search_kubeflow_docs("test query")

        assert result == "No results found for your query."

    def test_returns_formatted_results(self, inject_mocks, sample_milvus_hits):
        """Should return markdown-formatted results with scores and citations."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_milvus_hits

        result = server.search_kubeflow_docs("KServe")

        assert "Result 1" in result
        assert "Result 2" in result
        assert "0.9234" in result
        assert "https://www.kubeflow.org/docs/kserve/" in result
        assert "KServe provides serverless inference" in result

    def test_includes_file_path_in_results(self, inject_mocks, sample_milvus_hits):
        """Result should include the file path from Milvus."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_milvus_hits

        result = server.search_kubeflow_docs("KServe")

        assert "content/en/docs/kserve/overview.md" in result

    def test_respects_top_k_parameter(self, inject_mocks):
        """top_k should be passed through to Milvus client.search limit."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_kubeflow_docs("test", top_k=3)

        assert mock_client.search.call_args.kwargs["limit"] == 3

    def test_encodes_query_with_model(self, inject_mocks):
        """Should encode the query string using the sentence transformer."""
        mock_client, mock_model = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_kubeflow_docs("KServe setup guide")

        mock_model.encode.assert_called_once_with("KServe setup guide")

    def test_passes_embedding_to_milvus_as_list(self, inject_mocks):
        """Should pass the encoded embedding as a plain list to Milvus search."""
        mock_client, mock_model = inject_mocks
        fake_embedding = np.ones(768, dtype=np.float32)
        mock_model.encode.return_value = fake_embedding
        mock_client.search.return_value = [[]]

        server.search_kubeflow_docs("test")

        data = mock_client.search.call_args.kwargs["data"]
        assert len(data) == 1  # single query vector
        assert isinstance(data[0], list), "embedding must be a plain list"
        assert len(data[0]) == 768
        assert data[0][0] == 1.0  # from np.ones

    def test_requests_correct_output_fields(self, inject_mocks):
        """Should request content_text, citation_url, and file_path from Milvus."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_kubeflow_docs("test")

        output_fields = mock_client.search.call_args.kwargs["output_fields"]
        assert "content_text" in output_fields
        assert "citation_url" in output_fields
        assert "file_path" in output_fields

    def test_searches_correct_collection(self, inject_mocks):
        """Should search the configured COLLECTION_NAME."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_kubeflow_docs("test")

        assert mock_client.search.call_args.kwargs["collection_name"] == server.COLLECTION_NAME

    def test_handles_missing_entity_fields_gracefully(self, inject_mocks):
        """Should handle results where entity fields are missing without crashing."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[
            {
                "id": 1,
                "distance": 0.5,
                "entity": {},  # no fields
            }
        ]]

        result = server.search_kubeflow_docs("test")

        assert "Result 1" in result
        assert "0.5000" in result

    def test_results_separated_by_divider(self, inject_mocks, sample_milvus_hits):
        """Multiple results should be separated by --- dividers."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_milvus_hits

        result = server.search_kubeflow_docs("test")

        assert "\n---\n" in result

    def test_default_top_k_is_five(self, inject_mocks):
        """Default top_k should be 5 when not specified."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_kubeflow_docs("test")

        assert mock_client.search.call_args.kwargs["limit"] == 5


class TestSearchCollection:
    """Tests for the _search_collection shared helper."""

    def test_returns_empty_list_when_no_results(self, inject_mocks):
        """Should return empty list when Milvus returns no hits."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        result = server._search_collection(
            collection_name="test_col",
            query="test",
            top_k=5,
            output_fields=["content_text", "citation_url"],
        )
        assert result == []

    def test_passes_filter_expr_to_milvus(self, inject_mocks):
        """Should pass filter expression to Milvus search when provided."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server._search_collection(
            collection_name="test_col",
            query="test",
            top_k=5,
            output_fields=["content_text", "citation_url"],
            filter_expr='repo_name == "kubeflow/kubeflow"',
        )

        assert mock_client.search.call_args.kwargs["filter"] == 'repo_name == "kubeflow/kubeflow"'

    def test_omits_filter_when_empty(self, inject_mocks):
        """Should not include filter key when filter_expr is empty."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server._search_collection(
            collection_name="test_col",
            query="test",
            top_k=5,
            output_fields=["content_text", "citation_url"],
            filter_expr="",
        )

        assert "filter" not in mock_client.search.call_args.kwargs

    def test_returns_raw_hits_with_entity_data(self, inject_mocks):
        """Should return raw Milvus hits with entity data intact."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[{
            "id": 1,
            "distance": 0.9,
            "entity": {
                "content_text": "Test content",
                "citation_url": "https://example.com",
                "issue_number": 42,
            },
        }]]

        result = server._search_collection(
            collection_name="test_col",
            query="test",
            top_k=5,
            output_fields=["content_text", "citation_url", "issue_number"],
        )

        assert len(result) == 1
        assert result[0]["entity"]["issue_number"] == 42
        assert result[0]["entity"]["content_text"] == "Test content"
        assert result[0]["distance"] == 0.9


class TestSearchGithubIssues:
    """Tests for the search_github_issues MCP tool."""

    def test_returns_no_results_when_empty(self, inject_mocks):
        """Should return 'No issues found' when no issues match."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        result = server.search_github_issues("GPU OOM error")
        assert result == "No issues found for your query."

    def test_returns_formatted_results(self, inject_mocks, sample_issues_milvus_hits):
        """Should return formatted results with issue-specific fields."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_issues_milvus_hits

        result = server.search_github_issues("KServe model loading")

        assert "Result 1" in result
        assert "0.8912" in result
        assert "github.com/kubeflow/kubeflow/issues/42" in result
        assert "KServe model not loading" in result

    def test_includes_issue_number(self, inject_mocks, sample_issues_milvus_hits):
        """Should include issue number in formatted output."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_issues_milvus_hits

        result = server.search_github_issues("test")
        assert "**Issue:** #42" in result

    def test_includes_issue_labels(self, inject_mocks, sample_issues_milvus_hits):
        """Should include issue_labels in formatted output."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_issues_milvus_hits

        result = server.search_github_issues("test")
        assert "kind/bug, area/kserve" in result

    def test_filters_by_repo(self, inject_mocks):
        """Should construct repo filter expression."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_github_issues("test", repo="kubeflow/pipelines")

        filter_val = mock_client.search.call_args.kwargs.get("filter", "")
        assert 'repo_name == "kubeflow/pipelines"' in filter_val

    def test_filters_by_state(self, inject_mocks):
        """Should construct state filter expression."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_github_issues("test", state="open")

        filter_val = mock_client.search.call_args.kwargs.get("filter", "")
        assert 'issue_state == "open"' in filter_val

    def test_filters_by_repo_and_state(self, inject_mocks):
        """Should combine repo and state filters with 'and'."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_github_issues("test", repo="kubeflow/kubeflow", state="closed")

        filter_val = mock_client.search.call_args.kwargs["filter"]
        assert "repo_name" in filter_val
        assert "issue_state" in filter_val
        assert " and " in filter_val

    def test_no_filter_when_params_empty(self, inject_mocks):
        """Should not include filter when repo and state are empty."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_github_issues("test")

        assert "filter" not in mock_client.search.call_args.kwargs

    def test_searches_issues_collection(self, inject_mocks):
        """Should search the ISSUES_COLLECTION_NAME."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_github_issues("test")

        assert mock_client.search.call_args.kwargs["collection_name"] == server.ISSUES_COLLECTION_NAME

    def test_default_top_k_is_five(self, inject_mocks):
        """Default top_k should be 5."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_github_issues("test")

        assert mock_client.search.call_args.kwargs["limit"] == 5
