"""Tests for the MCP server (docs-agent-mcp/mcp-server/server.py).

Mocks pymilvus and embeddings HTTP calls — no in-process sentence-transformers.
"""

import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

MCP_SERVER_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "mcp-server"
MCP_SERVER_PATH = MCP_SERVER_DIR / "server.py"

sys.modules.setdefault("pymilvus", MagicMock())

sys.path.insert(0, str(MCP_SERVER_DIR))
spec = importlib.util.spec_from_file_location("docs_agent_mcp_server", MCP_SERVER_PATH)
server = importlib.util.module_from_spec(spec)
sys.modules["docs_agent_mcp_server"] = server
spec.loader.exec_module(server)


@pytest.fixture(autouse=True)
def reset_server_globals():
    """Reset server globals before each test so state doesn't leak."""
    original_client = server.client
    original_password = server.MILVUS_PASSWORD
    server.MILVUS_PASSWORD = "test-password"
    yield
    server.client = original_client
    server.MILVUS_PASSWORD = original_password


@pytest.fixture
def inject_mocks(mock_milvus_client):
    """Inject mock Milvus client and fixed query embedding."""
    server.client = mock_milvus_client
    fake_vector = [0.0] * 768
    with patch.object(server, "embed_query", return_value=fake_vector) as embed_mock:
        yield mock_milvus_client, embed_mock


class TestInit:
    """Tests for the _init() lazy initialization function."""

    def test_init_requires_milvus_password(self):
        server.client = None
        server.MILVUS_PASSWORD = ""
        with pytest.raises(RuntimeError, match="MILVUS_PASSWORD"):
            server._init()

    def test_init_creates_client_when_none(self):
        server.client = None
        server.MILVUS_PASSWORD = "secret"
        mock_mc_class = MagicMock(return_value=MagicMock())
        server.MilvusClient = mock_mc_class

        server._init()

        mock_mc_class.assert_called_once_with(
            uri=server.MILVUS_URI,
            user=server.MILVUS_USER,
            password="secret",
        )

    def test_init_is_idempotent(self):
        server.client = None
        server.MILVUS_PASSWORD = "secret"
        mock_mc_class = MagicMock(return_value=MagicMock())
        server.MilvusClient = mock_mc_class

        server._init()
        server._init()

        mock_mc_class.assert_called_once()


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

    def test_calls_embeddings_service_for_query(self, inject_mocks):
        mock_client, embed_mock = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_kubeflow_docs("KServe setup guide")

        embed_mock.assert_called_once()
        assert embed_mock.call_args[0][0] == "KServe setup guide"

    def test_passes_embedding_to_milvus(self, inject_mocks):
        mock_client, embed_mock = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_kubeflow_docs("test")

        data = mock_client.search.call_args.kwargs["data"]
        assert len(data) == 1
        assert len(data[0]) == 768

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
        mock_client.search.return_value = [
            [
                {
                    "id": 1,
                    "distance": 0.5,
                    "entity": {},  # no fields
                }
            ]
        ]

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
        mock_client.search.return_value = [
            [
                {
                    "id": 1,
                    "distance": 0.9,
                    "entity": {
                        "content_text": "Test content",
                        "citation_url": "https://example.com",
                        "issue_number": 42,
                    },
                }
            ]
        ]

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

    @pytest.mark.parametrize(
        ("field_name", "kwargs"),
        [
            ("repo", {"repo": 'kubeflow/pipelines" or issue_state == "open'}),
            ("state", {"state": 'open" or repo_name == "kubeflow/kubeflow'}),
        ],
    )
    def test_rejects_unsafe_filter_values(self, inject_mocks, field_name, kwargs):
        """User-controlled issue filters should not be interpolated unchecked."""
        mock_client, _ = inject_mocks

        with pytest.raises(ValueError, match=f"Invalid {field_name} filter value"):
            server.search_github_issues("test", **kwargs)

        mock_client.search.assert_not_called()


class TestSearchKubeflowCode:
    """Tests for the search_kubeflow_code MCP tool."""

    def test_returns_no_results_when_empty(self, inject_mocks):
        """Should return 'No code results found' when code search is empty."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        result = server.search_kubeflow_code("deployment")

        assert result == "No code results found for your query."

    def test_returns_formatted_code_results(self, inject_mocks, sample_code_milvus_hits):
        """Should return code results with resource metadata and fenced content."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_code_milvus_hits

        result = server.search_kubeflow_code("pipeline deployment")

        assert "### Result 1 (score: 0.8123)" in result
        assert "https://github.com/kubeflow/manifests/blob/main/apps/pipeline/deployment.yaml" in result
        assert "**File:** apps/pipeline/deployment.yaml" in result
        assert "**Resource:** Deployment `ml-pipeline` (namespace: kubeflow)" in result
        assert "**Type:** yaml" in result
        assert "```\napiVersion: apps/v1\nkind: Deployment" in result

    def test_results_separated_by_divider(self, inject_mocks, sample_code_milvus_hits):
        """Multiple code results should be separated by markdown dividers."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_code_milvus_hits

        result = server.search_kubeflow_code("test")

        assert "\n---\n" in result

    def test_searches_code_collection(self, inject_mocks):
        """Should search the CODE_COLLECTION_NAME."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_kubeflow_code("test")

        assert mock_client.search.call_args.kwargs["collection_name"] == server.CODE_COLLECTION_NAME

    def test_default_top_k_is_five(self, inject_mocks):
        """Default top_k should be 5."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_kubeflow_code("test")

        assert mock_client.search.call_args.kwargs["limit"] == 5

    def test_respects_top_k_parameter(self, inject_mocks):
        """top_k should be passed through to Milvus client.search limit."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_kubeflow_code("test", top_k=2)

        assert mock_client.search.call_args.kwargs["limit"] == 2

    def test_requests_code_output_fields(self, inject_mocks):
        """Should request code-specific output fields from Milvus."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_kubeflow_code("test")

        output_fields = mock_client.search.call_args.kwargs["output_fields"]
        assert "content_text" in output_fields
        assert "citation_url" in output_fields
        assert "file_path" in output_fields
        assert "resource_kind" in output_fields
        assert "resource_name" in output_fields
        assert "resource_namespace" in output_fields
        assert "file_type" in output_fields

    def test_filters_by_resource_kind(self, inject_mocks):
        """Should construct a resource_kind filter expression."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_kubeflow_code("test", resource_kind="Deployment")

        assert mock_client.search.call_args.kwargs["filter"] == "resource_kind == 'Deployment'"

    def test_no_filter_when_resource_kind_empty(self, inject_mocks):
        """Should not include filter when resource_kind is empty."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_kubeflow_code("test")

        assert "filter" not in mock_client.search.call_args.kwargs

    def test_rejects_unsafe_resource_kind_filter(self, inject_mocks):
        """resource_kind should not allow expression injection."""
        mock_client, _ = inject_mocks

        with pytest.raises(ValueError, match="Invalid resource_kind filter value"):
            server.search_kubeflow_code("test", resource_kind="Deployment' or file_type == 'python")

        mock_client.search.assert_not_called()
