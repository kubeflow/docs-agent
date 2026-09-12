"""Tests for the MCP server (docs-agent-mcp/mcp-server/server.py).

Mocks pymilvus and embeddings HTTP calls — no in-process sentence-transformers.
"""

import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastmcp.tools import ToolResult

MCP_SERVER_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "mcp-server"
MCP_SERVER_PATH = MCP_SERVER_DIR / "server.py"

_pymilvus_before = sys.modules.get("pymilvus")
sys.modules["pymilvus"] = MagicMock()
try:
    sys.path.insert(0, str(MCP_SERVER_DIR))
    spec = importlib.util.spec_from_file_location("docs_agent_mcp_server", MCP_SERVER_PATH)
    server = importlib.util.module_from_spec(spec)
    sys.modules["docs_agent_mcp_server"] = server
    spec.loader.exec_module(server)
finally:
    if _pymilvus_before is None:
        sys.modules.pop("pymilvus", None)
    else:
        sys.modules["pymilvus"] = _pymilvus_before

import intent_router  # noqa: E402
import milvus_search  # noqa: E402
from rag_collections import DENSE_FIELD, DOCS_COLLECTION, SPARSE_FIELD  # noqa: E402

MCP_MANIFEST_PATH = Path(__file__).parent.parent / "docs-agent-mcp" / "manifests" / "mcp-server" / "mcp-server.yaml"


def _tool_text(result: ToolResult | str) -> str:
    if isinstance(result, str):
        return result
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


def _tool_structured(result: ToolResult | str) -> dict | None:
    if isinstance(result, str):
        return None
    return result.structured_content


def _assert_no_urls_in_evidence(text: str) -> None:
    assert "https://" not in text
    assert "http://" not in text
    assert "**Source:**" not in text
    assert "```json" not in text


DOCS_FIELD_NAMES = [
    "id",
    "document_id",
    "content_text",
    "vector",
    "sparse_vector",
    "chunk_index",
    "citation_url",
    "file_path",
    "title",
    "section_path",
    "doc_type",
    "version",
    "release_date",
]

DENSE_ONLY_FIELD_NAMES = [
    "id",
    "content_text",
    "citation_url",
    "file_path",
    "vector",
]


def _schema_payload(*, field_names: list[str]) -> dict:
    return {"fields": [{"name": name} for name in field_names]}


def mock_docs_schema(mock_client, collection_name: str = DOCS_COLLECTION) -> None:
    """Collection has sparse_vector and release_date."""
    payload = _schema_payload(field_names=DOCS_FIELD_NAMES)
    mock_client.describe_collection.return_value = payload
    milvus_search._schema_cache[collection_name] = payload


def mock_dense_only_schema(mock_client, collection_name: str = DOCS_COLLECTION) -> None:
    """Collection has no sparse_vector (dense search only)."""
    payload = _schema_payload(field_names=DENSE_ONLY_FIELD_NAMES)
    mock_client.describe_collection.return_value = payload
    milvus_search._schema_cache[collection_name] = payload


@pytest.fixture(autouse=True)
def reset_search_globals():
    original_client = milvus_search.client
    original_password = milvus_search.MILVUS_PASSWORD
    original_local_mode = milvus_search.MILVUS_LOCAL_MODE
    original_search_mode = milvus_search.SEARCH_MODE
    original_issues_search_mode = milvus_search.ISSUES_SEARCH_MODE
    original_code_search_mode = milvus_search.CODE_SEARCH_MODE
    original_collection_name = milvus_search.COLLECTION_NAME
    original_schema_cache = dict(milvus_search._schema_cache)
    milvus_search.MILVUS_PASSWORD = "test-password"
    milvus_search.MILVUS_LOCAL_MODE = False
    milvus_search.SEARCH_MODE = "dense"
    milvus_search.ISSUES_SEARCH_MODE = "dense"
    milvus_search.CODE_SEARCH_MODE = "dense"
    milvus_search.COLLECTION_NAME = DOCS_COLLECTION
    milvus_search._schema_cache.clear()
    yield
    milvus_search.client = original_client
    milvus_search.MILVUS_PASSWORD = original_password
    milvus_search.MILVUS_LOCAL_MODE = original_local_mode
    milvus_search.SEARCH_MODE = original_search_mode
    milvus_search.ISSUES_SEARCH_MODE = original_issues_search_mode
    milvus_search.CODE_SEARCH_MODE = original_code_search_mode
    milvus_search.COLLECTION_NAME = original_collection_name
    milvus_search._schema_cache.clear()
    milvus_search._schema_cache.update(original_schema_cache)


@pytest.fixture
def inject_mocks(mock_milvus_client):
    milvus_search.client = mock_milvus_client
    fake_vector = [0.0] * 768
    with patch.object(milvus_search, "embed_query", return_value=fake_vector) as embed_mock:
        yield mock_milvus_client, embed_mock


class TestConnect:
    def test_connect_requires_milvus_password(self):
        milvus_search.client = None
        milvus_search.MILVUS_PASSWORD = ""
        with pytest.raises(RuntimeError, match="MILVUS_PASSWORD"):
            milvus_search.connect()

    def test_connect_creates_client_when_none(self):
        milvus_search.client = None
        milvus_search.MILVUS_PASSWORD = "secret"
        mock_mc_class = MagicMock(return_value=MagicMock())
        with patch.object(milvus_search, "MilvusClient", mock_mc_class):
            milvus_search.connect()

        mock_mc_class.assert_called_once_with(
            uri=milvus_search.MILVUS_URI,
            user=milvus_search.MILVUS_USER,
            password="secret",
        )

    def test_connect_is_idempotent(self):
        milvus_search.client = None
        milvus_search.MILVUS_PASSWORD = "secret"
        mock_mc_class = MagicMock(return_value=MagicMock())
        with patch.object(milvus_search, "MilvusClient", mock_mc_class):
            milvus_search.connect()
            milvus_search.connect()

        mock_mc_class.assert_called_once()

    def test_connect_allows_empty_password_in_local_mode(self):
        milvus_search.client = None
        milvus_search.MILVUS_PASSWORD = ""
        milvus_search.MILVUS_LOCAL_MODE = True
        mock_mc_class = MagicMock(return_value=MagicMock())
        with patch.object(milvus_search, "MilvusClient", mock_mc_class):
            milvus_search.connect()

        mock_mc_class.assert_called_once_with(
            uri=milvus_search.MILVUS_URI,
            user=milvus_search.MILVUS_USER,
            password="",
        )


class TestSearchKubeflowDocs:
    """Tests for the search_kubeflow_docs MCP tool."""

    def test_returns_no_results_message_when_empty(self, inject_mocks):
        """Should return 'No results found' when Milvus returns empty."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        result = server.search_kubeflow_docs("test query")

        assert isinstance(result, ToolResult)
        assert _tool_text(result) == "No results found for your query."
        assert _tool_structured(result) is None

    def test_returns_formatted_results(self, inject_mocks, sample_milvus_hits):
        """Should return markdown evidence plus structured citations."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_milvus_hits

        result = server.search_kubeflow_docs("KServe")
        text = _tool_text(result)
        structured = _tool_structured(result)

        assert "Result 1 [c1]" in text
        assert "Result 2 [c2]" in text
        assert "0.9234" in text
        assert "KServe provides serverless inference" in text
        _assert_no_urls_in_evidence(text)

        citations = structured["citations"]
        assert len(citations) == 2
        assert citations[0]["id"] == "c1"
        assert citations[0]["url"] == "https://www.kubeflow.org/docs/kserve/"
        assert citations[0]["file_path"] == "content/en/docs/kserve/overview.md"
        assert citations[1]["id"] == "c2"
        assert "retrieval" in structured

    def test_file_paths_only_in_structured_citations(self, inject_mocks, sample_milvus_hits):
        """File paths belong in citation metadata, not LLM-facing evidence."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_milvus_hits

        result = server.search_kubeflow_docs("KServe")
        text = _tool_text(result)
        citations = _tool_structured(result)["citations"]

        assert "content/en/docs/kserve/overview.md" not in text
        assert citations[0]["file_path"] == "content/en/docs/kserve/overview.md"

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

        assert mock_client.search.call_args.kwargs["collection_name"] == milvus_search.COLLECTION_NAME

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
        text = _tool_text(result)

        assert "Result 1 [c1]" in text
        assert "0.5000" in text
        _assert_no_urls_in_evidence(text)

    def test_results_separated_by_divider(self, inject_mocks, sample_milvus_hits):
        """Multiple results should be separated by --- dividers."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_milvus_hits

        result = server.search_kubeflow_docs("test")
        text = _tool_text(result)

        assert "\n---\n" in text

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

        result = milvus_search.search_collection(
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

        milvus_search.search_collection(
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

        milvus_search.search_collection(
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

        result = milvus_search.search_collection(
            collection_name="test_col",
            query="test",
            top_k=5,
            output_fields=["content_text", "citation_url", "issue_number"],
        )

        assert len(result) == 1
        assert result[0]["entity"]["issue_number"] == 42
        assert result[0]["entity"]["content_text"] == "Test content"
        assert result[0]["distance"] == 0.9

    def test_dense_mode_uses_client_search(self, inject_mocks):
        """Default dense mode should call MilvusClient.search, not hybrid_search."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        milvus_search.search_collection(
            collection_name="test_col",
            query="test",
            top_k=5,
            output_fields=["content_text"],
        )

        mock_client.search.assert_called_once()
        mock_client.hybrid_search.assert_not_called()

    def test_dense_mode_passes_anns_field(self, inject_mocks):
        """Dense search should target the configured dense vector field explicitly."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        milvus_search.search_collection(
            collection_name="test_col",
            query="test",
            top_k=5,
            output_fields=["content_text"],
        )

        assert mock_client.search.call_args.kwargs["anns_field"] == DENSE_FIELD

    def test_rejects_wrong_embedding_dimension(self, inject_mocks):
        """Should fail clearly when embedding length does not match HYBRID_DENSE_DIM."""
        mock_client, embed_mock = inject_mocks
        embed_mock.return_value = [0.0] * 512

        with pytest.raises(RuntimeError, match="Embedding dimension mismatch: expected 768, got 512"):
            milvus_search.search_collection(
                collection_name="test_col",
                query="test",
                top_k=5,
                output_fields=["content_text"],
            )

        mock_client.search.assert_not_called()
        mock_client.hybrid_search.assert_not_called()

    def test_hybrid_mode_builds_ann_requests_and_hybrid_search(self, inject_mocks):
        """Docs collection should use hybrid search when SEARCH_MODE=hybrid."""
        mock_client, embed_mock = inject_mocks
        milvus_search.SEARCH_MODE = "hybrid"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_docs_schema(mock_client, DOCS_COLLECTION)
        mock_client.hybrid_search.return_value = [[]]

        with (
            patch.object(milvus_search, "AnnSearchRequest") as request_class,
            patch.object(milvus_search, "RRFRanker") as ranker_class,
        ):
            milvus_search.search_collection(
                collection_name=milvus_search.COLLECTION_NAME,
                query="install pipelines",
                top_k=4,
                output_fields=["content_text", "citation_url", "file_path"],
            )

        embed_mock.assert_called_once_with("install pipelines", url=milvus_search.EMBEDDINGS_URL or None)
        mock_client.search.assert_not_called()
        mock_client.hybrid_search.assert_called_once()

        dense_call, sparse_call = request_class.call_args_list
        assert dense_call.kwargs["anns_field"] == DENSE_FIELD
        assert dense_call.kwargs["param"] == {"metric_type": "COSINE"}
        assert len(dense_call.kwargs["data"][0]) == 768
        assert dense_call.kwargs["limit"] == 4
        assert "expr" not in dense_call.kwargs

        assert sparse_call.kwargs["anns_field"] == SPARSE_FIELD
        assert sparse_call.kwargs["param"] == {"metric_type": "BM25"}
        assert sparse_call.kwargs["data"] == ["install pipelines"]
        assert sparse_call.kwargs["limit"] == 4

        hybrid_kwargs = mock_client.hybrid_search.call_args.kwargs
        assert hybrid_kwargs["collection_name"] == DOCS_COLLECTION
        assert hybrid_kwargs["limit"] == 4
        assert hybrid_kwargs["output_fields"] == ["content_text", "citation_url", "file_path"]
        assert len(hybrid_kwargs["reqs"]) == 2
        ranker_class.assert_called_once()

    def test_hybrid_mode_keeps_issues_dense(self, inject_mocks):
        """Issues collection should stay dense when only docs SEARCH_MODE=hybrid."""
        mock_client, _ = inject_mocks
        milvus_search.SEARCH_MODE = "hybrid"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_client.search.return_value = [[]]

        server.search_github_issues("GPU OOM error")

        mock_client.search.assert_called_once()
        mock_client.hybrid_search.assert_not_called()
        assert mock_client.search.call_args.kwargs["collection_name"] == milvus_search.ISSUES_COLLECTION_NAME
        assert mock_client.search.call_args.kwargs["anns_field"] == DENSE_FIELD

    def test_hybrid_mode_keeps_code_dense(self, inject_mocks):
        """Code collection should stay dense when only docs SEARCH_MODE=hybrid."""
        mock_client, _ = inject_mocks
        milvus_search.SEARCH_MODE = "hybrid"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_client.search.return_value = [[]]

        server.search_kubeflow_code("deployment")

        mock_client.search.assert_called_once()
        mock_client.hybrid_search.assert_not_called()
        assert mock_client.search.call_args.kwargs["collection_name"] == milvus_search.CODE_COLLECTION_NAME
        assert mock_client.search.call_args.kwargs["anns_field"] == DENSE_FIELD

    def test_hybrid_mode_propagates_filter_to_ann_requests(self, inject_mocks):
        """Hybrid AnnSearchRequest objects should receive the filter expression."""
        mock_client, _ = inject_mocks
        milvus_search.SEARCH_MODE = "hybrid"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_docs_schema(mock_client, DOCS_COLLECTION)
        mock_client.hybrid_search.return_value = [[]]
        filter_expr = 'repo_name == "kubeflow/pipelines"'

        with patch.object(milvus_search, "AnnSearchRequest") as request_class:
            milvus_search.search_collection(
                collection_name=milvus_search.COLLECTION_NAME,
                query="test",
                top_k=5,
                output_fields=["content_text"],
                filter_expr=filter_expr,
            )

        assert request_class.call_count == 2
        dense_call, sparse_call = request_class.call_args_list
        assert dense_call.kwargs["expr"] == filter_expr
        assert sparse_call.kwargs["expr"] == filter_expr
        assert "filter" not in mock_client.hybrid_search.call_args.kwargs

    def test_hybrid_mode_returns_search_failed_on_milvus_error(self, inject_mocks):
        """Tool layer should surface hybrid Milvus failures safely."""
        mock_client, _ = inject_mocks
        milvus_search.SEARCH_MODE = "hybrid"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_docs_schema(mock_client, DOCS_COLLECTION)
        mock_client.hybrid_search.side_effect = RuntimeError("bm25 unavailable")

        result = server.search_kubeflow_docs("test")
        text = _tool_text(result)

        assert isinstance(result, ToolResult)
        assert text.startswith("Search failed:")
        assert "hybrid_search failed" in text
        assert "bm25 unavailable" in text
        assert _tool_structured(result) is None

    def test_hybrid_mode_falls_back_to_dense_without_sparse_field(self, inject_mocks):
        """Hybrid mode should degrade to dense search when sparse_vector is absent."""
        mock_client, embed_mock = inject_mocks
        milvus_search.SEARCH_MODE = "hybrid"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_dense_only_schema(mock_client, DOCS_COLLECTION)
        mock_client.search.return_value = [[]]

        server.search_kubeflow_docs("install pipelines")

        embed_mock.assert_called_once()
        mock_client.search.assert_called_once()
        mock_client.hybrid_search.assert_not_called()
        assert mock_client.search.call_args.kwargs["collection_name"] == DOCS_COLLECTION
        assert mock_client.search.call_args.kwargs["anns_field"] == DENSE_FIELD


class TestCollectionFields:
    def test_docs_schema_has_bm25_and_release_date(self, inject_mocks):
        mock_client, _ = inject_mocks
        mock_docs_schema(mock_client, DOCS_COLLECTION)

        assert milvus_search.collection_has_bm25(DOCS_COLLECTION) is True
        assert milvus_search.collection_has_release_fields(DOCS_COLLECTION) is True

    def test_dense_only_schema_has_no_bm25(self, inject_mocks):
        mock_client, _ = inject_mocks
        mock_dense_only_schema(mock_client, DOCS_COLLECTION)

        assert milvus_search.collection_has_bm25(DOCS_COLLECTION) is False
        assert milvus_search.collection_has_release_fields(DOCS_COLLECTION) is False


class TestAutoDocsRouting:

    def test_production_name_temporal_uses_bm25_and_release_filter(self, inject_mocks):
        mock_client, embed_mock = inject_mocks
        milvus_search.SEARCH_MODE = "auto"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_docs_schema(mock_client, DOCS_COLLECTION)
        mock_client.search.return_value = [
            [
                {
                    "id": 1,
                    "distance": 0.88,
                    "entity": {
                        "content_text": "Kubeflow 1.9 release",
                        "citation_url": "https://example.com/1.9",
                        "file_path": "releases/kubeflow-1.9.md",
                        "doc_type": "release",
                        "release_date": 1721606400,
                    },
                }
            ]
        ]

        server.search_kubeflow_docs("latest Kubeflow release")

        embed_mock.assert_not_called()
        first_call = mock_client.search.call_args_list[0].kwargs
        assert first_call["collection_name"] == DOCS_COLLECTION
        assert first_call["anns_field"] == SPARSE_FIELD
        assert first_call["filter"] == 'doc_type == "release"'
        assert first_call["limit"] == intent_router.AUTO_TEMPORAL_CANDIDATE_DEPTH

    def test_auto_requests_release_output_fields(self, inject_mocks):
        mock_client, _ = inject_mocks
        milvus_search.SEARCH_MODE = "auto"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_docs_schema(mock_client, DOCS_COLLECTION)
        mock_client.search.return_value = [[]]

        server.search_kubeflow_docs("latest release")

        output_fields = mock_client.search.call_args.kwargs["output_fields"]
        assert "release_date" in output_fields
        assert "doc_type" in output_fields
        assert "version" in output_fields


class TestNoSparseFallback:
    """Collections without sparse_vector fall back to dense search."""

    def test_auto_temporal_falls_back_to_dense_without_sparse(self, inject_mocks):
        mock_client, embed_mock = inject_mocks
        milvus_search.SEARCH_MODE = "auto"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_dense_only_schema(mock_client, DOCS_COLLECTION)
        mock_client.search.return_value = [
            [
                {
                    "id": 1,
                    "distance": 0.77,
                    "entity": {
                        "content_text": "legacy dense hit",
                        "citation_url": "https://example.com",
                        "file_path": "doc.md",
                    },
                }
            ]
        ]

        result = server.search_kubeflow_docs("latest Kubeflow release")
        text = _tool_text(result)
        structured = _tool_structured(result)

        embed_mock.assert_called_once()
        kwargs = mock_client.search.call_args.kwargs
        assert kwargs["anns_field"] == DENSE_FIELD
        assert "filter" not in kwargs
        assert structured["retrieval"]["retrieval_mode"] == "dense"
        assert "dense fallback" in structured["retrieval"]["reason"]
        _assert_no_urls_in_evidence(text)
        mock_client.hybrid_search.assert_not_called()

    def test_auto_exact_falls_back_to_dense_without_sparse(self, inject_mocks):
        mock_client, embed_mock = inject_mocks
        milvus_search.SEARCH_MODE = "auto"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_dense_only_schema(mock_client, DOCS_COLLECTION)
        mock_client.search.return_value = [[]]

        server.search_kubeflow_docs("metadata.name field error")

        embed_mock.assert_called_once()
        assert mock_client.search.call_args.kwargs["anns_field"] == DENSE_FIELD

    def test_dense_mode_does_not_probe_schema(self, inject_mocks):
        mock_client, embed_mock = inject_mocks
        milvus_search.SEARCH_MODE = "dense"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_client.search.return_value = [[]]

        server.search_kubeflow_docs("KServe")

        embed_mock.assert_called_once()
        mock_client.describe_collection.assert_not_called()
        assert mock_client.search.call_args.kwargs["anns_field"] == DENSE_FIELD


class TestMcpServerConfigContract:
    """Production manifest exposes SEARCH_MODE=auto defaults."""

    def test_manifest_sets_auto_mode_and_router_tuning(self):
        import yaml

        documents = list(yaml.safe_load_all(MCP_MANIFEST_PATH.read_text(encoding="utf-8")))
        config_maps = [doc for doc in documents if doc.get("kind") == "ConfigMap"]
        assert config_maps, "expected mcp-server ConfigMap"
        data = config_maps[0]["data"]

        assert data["COLLECTION_NAME"] == DOCS_COLLECTION
        assert data["SEARCH_MODE"] == "auto"
        assert data["HYBRID_RANKER"] == "rrf"
        assert data["AUTO_TEMPORAL_CANDIDATE_DEPTH"] == "50"
        assert data["AUTO_BM25_CANDIDATE_DEPTH"] == "30"
        assert data["HYBRID_RRF_K"] == "60"


class TestQueryIntentClassification:
    """Deterministic docs query intent classification for SEARCH_MODE=auto."""

    @pytest.mark.parametrize(
        ("query", "expected_intent", "expected_mode"),
        [
            ("What is the latest Kubeflow release?", "temporal", "bm25"),
            ("current supported version", "temporal", "bm25"),
            ("newest release notes", "temporal", "bm25"),
            ("most recent GA version", "temporal", "bm25"),
            ("when was Kubeflow 1.9 released?", "release_date", "bm25"),
            ("Kubeflow 1.8 release date", "release_date", "bm25"),
            ("GA date for version 1.7", "release_date", "bm25"),
            ("metadata.name config key", "exact", "bm25"),
            ("apiVersion v1beta1 Deployment", "exact", "bm25"),
            ("CrashLoopBackOff error in pod", "exact", "bm25"),
            ("how does KServe architecture work?", "conceptual", "hybrid"),
            ("explain pipeline overview", "conceptual", "hybrid"),
            ("compare Kubeflow 1.8 and 1.9", "comparison", "hybrid"),
            ("differences between 1.7 vs 1.8", "comparison", "hybrid"),
            ("install Kubeflow pipelines", "general", "hybrid"),
        ],
    )
    def test_classify_query_intent(self, query, expected_intent, expected_mode):
        plan = intent_router.pick_search_plan(query)
        assert plan.intent == expected_intent
        assert plan.retrieval_mode == expected_mode

    def test_temporal_plan_requests_release_filter_and_rerank(self):
        plan = intent_router.pick_search_plan("latest supported Kubeflow version")
        assert plan.filter_expr == 'doc_type == "release"'
        assert plan.rerank_by_release_date is True
        assert plan.candidate_depth == intent_router.AUTO_TEMPORAL_CANDIDATE_DEPTH


class TestTemporalReleaseDateReranking:
    """Date-aware reranking for temporal BM25 candidates."""

    def test_rerank_by_release_date_orders_newest_first(self):
        hits = [
            {"distance": 0.95, "entity": {"release_date": 1564531200, "content_text": "1.0"}},
            {"distance": 0.99, "entity": {"release_date": 1721606400, "content_text": "1.9"}},
            {"distance": 0.97, "entity": {"release_date": 1693526400, "content_text": "1.8"}},
        ]
        reranked = intent_router.rerank_by_release_date(hits, top_k=2)
        assert [hit["entity"]["content_text"] for hit in reranked] == ["1.9", "1.8"]

    def test_rerank_by_release_date_tiebreaks_on_lexical_score(self):
        hits = [
            {"distance": 0.80, "entity": {"release_date": 1700000000, "content_text": "a"}},
            {"distance": 0.95, "entity": {"release_date": 1700000000, "content_text": "b"}},
        ]
        reranked = intent_router.rerank_by_release_date(hits, top_k=2)
        assert reranked[0]["entity"]["content_text"] == "b"

    def test_rerank_by_release_date_falls_back_without_dates(self):
        hits = [
            {"distance": 0.91, "entity": {"content_text": "first"}},
            {"distance": 0.82, "entity": {"content_text": "second"}},
        ]
        reranked = intent_router.rerank_by_release_date(hits, top_k=2)
        assert [hit["entity"]["content_text"] for hit in reranked] == ["first", "second"]

    def test_rerank_for_version_match_prefers_matching_version(self):
        hits = [
            {"distance": 0.99, "entity": {"version": "1.9", "content_text": "newest"}},
            {"distance": 0.85, "entity": {"version": "1.8", "content_text": "target"}},
        ]
        reranked = intent_router.rerank_for_version_match(hits, "when was Kubeflow 1.8 released?", top_k=1)
        assert reranked[0]["entity"]["content_text"] == "target"

    def test_boost_release_docs_prefers_release_doc_type(self):
        hits = [
            {"distance": 0.99, "entity": {"doc_type": "documentation", "content_text": "doc"}},
            {"distance": 0.80, "entity": {"doc_type": "release", "content_text": "release"}},
        ]
        boosted = intent_router.boost_release_docs(hits)
        assert boosted[0]["entity"]["content_text"] == "release"


class TestAutoSearchRouting:
    """SEARCH_MODE=auto docs retrieval routing."""

    def test_auto_temporal_uses_bm25_with_depth_and_metadata(self, inject_mocks):
        mock_client, embed_mock = inject_mocks
        milvus_search.SEARCH_MODE = "auto"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_docs_schema(mock_client, DOCS_COLLECTION)
        mock_client.search.return_value = [
            [
                {
                    "id": 1,
                    "distance": 0.88,
                    "entity": {
                        "content_text": "Kubeflow 1.9 release",
                        "citation_url": "https://example.com/1.9",
                        "file_path": "releases/kubeflow-1.9.md",
                        "doc_type": "release",
                        "release_date": 1721606400,
                        "version": "1.9",
                        "section_path": "Releases > 1.9",
                    },
                }
            ]
        ]

        result = server.search_kubeflow_docs("latest Kubeflow release")
        text = _tool_text(result)
        structured = _tool_structured(result)

        embed_mock.assert_not_called()
        kwargs = mock_client.search.call_args.kwargs
        assert kwargs["anns_field"] == SPARSE_FIELD
        assert kwargs["data"] == ["latest Kubeflow release"]
        assert kwargs["limit"] == intent_router.AUTO_TEMPORAL_CANDIDATE_DEPTH
        assert kwargs["filter"] == 'doc_type == "release"'
        assert "Result 1 [c1]" in text
        assert structured["retrieval"]["intent"] == "temporal"
        assert structured["retrieval"]["retrieval_mode"] == "bm25"
        _assert_no_urls_in_evidence(text)
        mock_client.hybrid_search.assert_not_called()

    def test_auto_temporal_retries_without_filter_when_empty(self, inject_mocks):
        mock_client, _ = inject_mocks
        milvus_search.SEARCH_MODE = "auto"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_docs_schema(mock_client, DOCS_COLLECTION)
        mock_client.search.side_effect = [
            [[]],
            [
                [
                    {
                        "id": 1,
                        "distance": 0.75,
                        "entity": {
                            "content_text": "fallback",
                            "citation_url": "https://example.com",
                            "file_path": "doc.md",
                        },
                    }
                ]
            ],
        ]

        result = server.search_kubeflow_docs("newest release")
        text = _tool_text(result)
        structured = _tool_structured(result)

        assert mock_client.search.call_count == 2
        assert mock_client.search.call_args_list[0].kwargs["filter"] == 'doc_type == "release"'
        assert "filter" not in mock_client.search.call_args_list[1].kwargs
        assert "fallback" in text
        assert structured["retrieval"]["filter_fallback"] is True
        _assert_no_urls_in_evidence(text)

    def test_auto_conceptual_uses_hybrid(self, inject_mocks):
        mock_client, embed_mock = inject_mocks
        milvus_search.SEARCH_MODE = "auto"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_docs_schema(mock_client, DOCS_COLLECTION)
        mock_client.hybrid_search.return_value = [[]]

        server.search_kubeflow_docs("how does KServe work?")

        embed_mock.assert_called_once()
        mock_client.hybrid_search.assert_called_once()
        mock_client.search.assert_not_called()

    def test_auto_exact_uses_bm25_without_embedding(self, inject_mocks):
        mock_client, embed_mock = inject_mocks
        milvus_search.SEARCH_MODE = "auto"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_docs_schema(mock_client, DOCS_COLLECTION)
        mock_client.search.return_value = [[]]

        server.search_kubeflow_docs("metadata.name field error")

        embed_mock.assert_not_called()
        kwargs = mock_client.search.call_args.kwargs
        assert kwargs["anns_field"] == SPARSE_FIELD
        assert kwargs["limit"] == intent_router.AUTO_BM25_CANDIDATE_DEPTH

    def test_auto_without_sparse_uses_dense_fallback(self, inject_mocks):
        mock_client, embed_mock = inject_mocks
        milvus_search.SEARCH_MODE = "auto"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_dense_only_schema(mock_client, DOCS_COLLECTION)
        mock_client.search.return_value = [[]]

        server.search_kubeflow_docs("latest release")

        embed_mock.assert_called_once()
        assert mock_client.search.call_args.kwargs["anns_field"] == DENSE_FIELD
        assert "filter" not in mock_client.search.call_args.kwargs

    def test_auto_includes_release_fields_in_output(self, inject_mocks):
        mock_client, _ = inject_mocks
        milvus_search.SEARCH_MODE = "auto"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_docs_schema(mock_client, DOCS_COLLECTION)
        mock_client.search.return_value = [
            [
                {
                    "id": 1,
                    "distance": 0.77,
                    "entity": {
                        "content_text": "release chunk",
                        "citation_url": "https://example.com",
                        "file_path": "releases/kubeflow-1.9.md",
                        "version": "1.9",
                        "section_path": "Releases > 1.9",
                        "release_date": 1721606400,
                    },
                }
            ]
        ]

        result = server.search_kubeflow_docs("latest release")
        text = _tool_text(result)
        structured = _tool_structured(result)

        output_fields = mock_client.search.call_args.kwargs["output_fields"]
        assert "release_date" in output_fields
        assert "version" in output_fields
        assert "**Version:** 1.9" in text
        assert "**Release date:** 1721606400" in text
        assert structured["citations"][0]["version"] == "1.9"
        assert structured["citations"][0]["release_date"] == 1721606400
        _assert_no_urls_in_evidence(text)


class TestExplicitSearchModeBackwardCompatibility:
    """Explicit dense/hybrid SEARCH_MODE behavior is unchanged."""

    def test_dense_mode_unchanged_without_metadata_block(self, inject_mocks, sample_milvus_hits):
        mock_client, _ = inject_mocks
        milvus_search.SEARCH_MODE = "dense"
        mock_client.search.return_value = sample_milvus_hits

        result = server.search_kubeflow_docs("KServe")
        text = _tool_text(result)
        structured = _tool_structured(result)

        assert "Result 1 [c1]" in text
        assert "0.9234" in text
        assert "```json" not in text
        assert "retrieval" in structured
        mock_client.hybrid_search.assert_not_called()

    def test_hybrid_mode_unchanged_without_metadata_block(self, inject_mocks):
        mock_client, _ = inject_mocks
        milvus_search.SEARCH_MODE = "hybrid"
        milvus_search.COLLECTION_NAME = DOCS_COLLECTION
        mock_docs_schema(mock_client, DOCS_COLLECTION)
        mock_client.hybrid_search.return_value = [
            [
                {
                    "id": 1,
                    "distance": 0.91,
                    "entity": {
                        "content_text": "hybrid hit",
                        "citation_url": "https://example.com",
                        "file_path": "doc.md",
                    },
                }
            ]
        ]

        result = server.search_kubeflow_docs("install pipelines")
        text = _tool_text(result)
        structured = _tool_structured(result)

        assert "hybrid hit" in text
        assert "```json" not in text
        assert structured["retrieval"]["retrieval_mode"] == "hybrid"
        _assert_no_urls_in_evidence(text)
        mock_client.hybrid_search.assert_called_once()
        mock_client.search.assert_not_called()


class TestSearchGithubIssues:
    """Tests for the search_github_issues MCP tool."""

    def test_returns_no_results_when_empty(self, inject_mocks):
        """Should return 'No issues found' when no issues match."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        result = server.search_github_issues("GPU OOM error")
        assert isinstance(result, ToolResult)
        assert _tool_text(result) == "No issues found for your query."
        assert _tool_structured(result) is None

    def test_returns_formatted_results(self, inject_mocks, sample_issues_milvus_hits):
        """Should return formatted evidence plus structured issue citations."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_issues_milvus_hits

        result = server.search_github_issues("KServe model loading")
        text = _tool_text(result)
        structured = _tool_structured(result)

        assert "Result 1 [c1]" in text
        assert "0.8912" in text
        assert "KServe model not loading" in text
        _assert_no_urls_in_evidence(text)

        citations = structured["citations"]
        assert citations[0]["id"] == "c1"
        assert citations[0]["url"] == "https://github.com/kubeflow/kubeflow/issues/42"
        assert citations[0]["issue_number"] == 42

    def test_includes_issue_number(self, inject_mocks, sample_issues_milvus_hits):
        """Should include issue number in formatted output."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_issues_milvus_hits

        result = server.search_github_issues("test")
        text = _tool_text(result)
        assert "**Issue:** #42" in text
        _assert_no_urls_in_evidence(text)

    def test_includes_issue_labels(self, inject_mocks, sample_issues_milvus_hits):
        """Should include issue_labels in formatted output."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_issues_milvus_hits

        result = server.search_github_issues("test")
        text = _tool_text(result)
        assert "kind/bug, area/kserve" in text
        assert _tool_structured(result)["citations"][0]["issue_labels"] == "kind/bug, area/kserve"

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

        assert mock_client.search.call_args.kwargs["collection_name"] == milvus_search.ISSUES_COLLECTION_NAME

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
        assert isinstance(result, ToolResult)
        assert _tool_text(result) == "No code results found for your query."
        assert _tool_structured(result) is None

    def test_returns_formatted_code_results(self, inject_mocks, sample_code_milvus_hits):
        """Should return code evidence plus structured code citations."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_code_milvus_hits

        result = server.search_kubeflow_code("pipeline deployment")
        text = _tool_text(result)
        structured = _tool_structured(result)

        assert "### Result 1 [c1] (score: 0.8123)" in text
        assert "**Resource:** Deployment `ml-pipeline` (namespace: kubeflow)" in text
        assert "**Type:** yaml" in text
        assert "```\napiVersion: apps/v1\nkind: Deployment" in text
        _assert_no_urls_in_evidence(text)
        assert "apps/pipeline/deployment.yaml" not in text

        citations = structured["citations"]
        assert citations[0]["id"] == "c1"
        assert citations[0]["url"] == (
            "https://github.com/kubeflow/manifests/blob/main/apps/pipeline/deployment.yaml"
        )
        assert citations[0]["file_path"] == "apps/pipeline/deployment.yaml"

    def test_results_separated_by_divider(self, inject_mocks, sample_code_milvus_hits):
        """Multiple code results should be separated by markdown dividers."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = sample_code_milvus_hits

        result = server.search_kubeflow_code("test")
        text = _tool_text(result)

        assert "\n---\n" in text
        _assert_no_urls_in_evidence(text)

    def test_searches_code_collection(self, inject_mocks):
        """Should search the CODE_COLLECTION_NAME."""
        mock_client, _ = inject_mocks
        mock_client.search.return_value = [[]]

        server.search_kubeflow_code("test")

        assert mock_client.search.call_args.kwargs["collection_name"] == milvus_search.CODE_COLLECTION_NAME

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
