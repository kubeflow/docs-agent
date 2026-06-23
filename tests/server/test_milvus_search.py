#Tests for milvus_search() function in app.py

import pytest
from importlib import import_module


@pytest.fixture(autouse=True)
def _load_app(env_vars):
    """Ensure env vars are set before the server module is referenced"""
    pass


def _get_milvus_search():
    mod = import_module("server-https.app")
    return mod.milvus_search


class TestMilvusSearchHappyPath:

    def test_returns_results_with_correct_keys(self, patch_encoder, patch_milvus):
        result = _get_milvus_search()("kubeflow pipelines")
        hits = result["results"]
        assert len(hits) == 2
        for hit in hits:
            assert set(hit.keys()) == {"similarity", "file_path", "citation_url", "content_text"}

    def test_similarity_is_one_minus_distance(self, patch_encoder, patch_milvus):
        result = _get_milvus_search()("kubeflow pipelines")
        hits = result["results"]
        # first hit has distance=0.12, so similarity should be 0.88
        assert hits[0]["similarity"] == pytest.approx(0.88, abs=1e-6)
        # second hit has distance=0.25, so similarity should be 0.75
        assert hits[1]["similarity"] == pytest.approx(0.75, abs=1e-6)

    def test_content_fields_match_fixture_data(self, patch_encoder, patch_milvus):
        result = _get_milvus_search()("kubeflow pipelines")
        hits = result["results"]
        assert hits[0]["file_path"] == "docs/pipelines/overview.md"
        assert "platform for building" in hits[0]["content_text"]
        assert hits[1]["citation_url"] == "https://www.kubeflow.org/docs/components/pipelines/sdk/"

    def test_calls_milvus_with_correct_parameters(self, patch_encoder, patch_milvus, mock_encoder):
        _get_milvus_search()("test query", top_k=7)

        # encoder called with the query string
        mock_encoder.encode.assert_called_once_with("test query")

        # collection.search is called with the right limit
        collection = patch_milvus["instance"]
        call_kwargs = collection.search.call_args
        assert call_kwargs.kwargs["limit"] == 7
        assert call_kwargs.kwargs["anns_field"] == "vector"


class TestMilvusSearchTopK:

    def test_custom_top_k_passed_to_collection_search(self, patch_encoder, patch_milvus):
        _get_milvus_search()("query", top_k=3)
        collection = patch_milvus["instance"]
        assert collection.search.call_args.kwargs["limit"] == 3


class TestMilvusSearchNoResults:

    def test_empty_results_when_collection_returns_nothing(self, patch_encoder, patch_milvus):
        patch_milvus["instance"].search.return_value = [[]]
        result = _get_milvus_search()("obscure query")
        assert result == {"results": []}


class TestMilvusSearchErrorHandling:

    def test_returns_empty_on_encoder_exception(self, patch_milvus):
        """If SentenceTransformer raises, milvus_search catches it and returns empty results."""
        from unittest.mock import patch, MagicMock

        broken_encoder = MagicMock()
        broken_encoder.encode.side_effect = RuntimeError("model load failed")
        with patch("server-https.app.SentenceTransformer", return_value=broken_encoder):
            result = _get_milvus_search()("anything")
        assert result == {"results": []}

    def test_returns_empty_on_connection_failure(self, patch_encoder):
        #If Milvus connection fails, milvus_search catches it and returns the empty results
        from unittest.mock import patch

        with patch("server-https.app.connections") as mock_conn, \
             patch("server-https.app.Collection"):
            mock_conn.connect.side_effect = ConnectionError("milvus down")
            result = _get_milvus_search()("anything")
        assert result == {"results": []}


class TestMilvusSearchContentTruncation:

    def test_long_content_is_truncated_at_400_chars(self, patch_encoder, patch_milvus):
        #content_text longer than 400 chars gets clipped with '...' for indicating truncation 
        from tests.conftest import _make_milvus_hit

        long_text = "A" * 500
        long_hit = _make_milvus_hit(
            file_path="docs/long.md",
            content_text=long_text,
            citation_url="https://example.com",
            distance=0.1,
        )
        patch_milvus["instance"].search.return_value = [[long_hit]]

        result = _get_milvus_search()("query")
        content = result["results"][0]["content_text"]
        assert len(content) == 403  # 400 chars + "..."
        assert content.endswith("...")
