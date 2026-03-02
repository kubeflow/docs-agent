"""Tests for the Kagent-native MCP server tools.

These tests mock pymilvus, sentence_transformers, and shared.rag_core so
they can run without Milvus or GPU hardware.  They verify:

- Partition routing (each tool hits the right partition)
- Quality-aware broadening (low similarity triggers a retry)
- Result formatting (Markdown output)
- Error handling (Milvus failures return graceful messages)
- Thread-safe encoder singleton
- Edge cases (empty results, max retries, clamped top_k)
"""

import threading
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# We need to mock heavy imports before importing the module under test.
# ---------------------------------------------------------------------------

def _make_mock_hit(similarity: float, file_path: str, citation_url: str, content_text: str):
    """Build a mock Milvus search hit."""
    hit = MagicMock()
    hit.distance = 1.0 - similarity  # COSINE: distance = 1 - similarity
    hit.entity.get = lambda key, default="": {
        "file_path": file_path,
        "citation_url": citation_url,
        "content_text": content_text,
    }.get(key, default)
    return hit


class TestFormatResults(unittest.TestCase):
    """Test _format_results() output formatting."""

    def setUp(self):
        # Import after patching is not needed here since _format_results
        # is a pure function that doesn't touch pymilvus.
        from agent.mcp_server import _format_results
        self._format_results = _format_results

    def test_empty_hits(self):
        result = self._format_results([])
        self.assertIn("No relevant results found", result)

    def test_single_hit_format(self):
        hits = [
            {
                "similarity": 0.85,
                "file_path": "docs/pipelines.md",
                "citation_url": "https://kubeflow.org/docs/pipelines",
                "content_text": "KFP lets you build ML pipelines.",
            }
        ]
        result = self._format_results(hits)
        self.assertIn("### Result 1", result)
        self.assertIn("0.8500", result)
        self.assertIn("https://kubeflow.org/docs/pipelines", result)
        self.assertIn("docs/pipelines.md", result)
        self.assertIn("KFP lets you build ML pipelines.", result)

    def test_multiple_hits_separated(self):
        hits = [
            {
                "similarity": 0.9,
                "file_path": "a.md",
                "citation_url": "https://a",
                "content_text": "Content A",
            },
            {
                "similarity": 0.7,
                "file_path": "b.md",
                "citation_url": "https://b",
                "content_text": "Content B",
            },
        ]
        result = self._format_results(hits)
        self.assertIn("### Result 1", result)
        self.assertIn("### Result 2", result)
        self.assertIn("---", result)

    def test_hit_without_optional_fields(self):
        hits = [
            {
                "similarity": 0.5,
                "file_path": None,
                "citation_url": None,
                "content_text": "Some text",
            }
        ]
        result = self._format_results(hits)
        self.assertIn("### Result 1", result)
        self.assertIn("Some text", result)
        # Should NOT contain **Source:** or **File:** if values are None
        self.assertNotIn("**Source:**", result)
        self.assertNotIn("**File:**", result)


class TestSearch(unittest.TestCase):
    """Test _search() partition routing and quality broadening."""

    @patch("agent.mcp_server._get_encoder")
    @patch("agent.mcp_server._ensure_milvus")
    @patch("agent.mcp_server.Collection")
    def test_search_uses_partition(self, mock_coll_cls, mock_ensure, mock_enc):
        """Verify that a non-empty partition name is passed to collection.search()."""
        from agent.mcp_server import _search

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 768)
        mock_enc.return_value = mock_encoder

        mock_collection = MagicMock()
        hit = _make_mock_hit(0.9, "docs/a.md", "https://a", "text")
        mock_collection.search.return_value = [[hit]]
        mock_coll_cls.return_value = mock_collection

        hits = _search("kubeflow pipeline", partition="platform_arch")

        # Verify partition_names was passed
        call_kwargs = mock_collection.search.call_args[1]
        self.assertEqual(call_kwargs["partition_names"], ["platform_arch"])
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0]["similarity"], 0.9, places=2)

    @patch("agent.mcp_server._get_encoder")
    @patch("agent.mcp_server._ensure_milvus")
    @patch("agent.mcp_server.Collection")
    def test_search_no_partition(self, mock_coll_cls, mock_ensure, mock_enc):
        """Empty partition string should NOT send partition_names."""
        from agent.mcp_server import _search

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 768)
        mock_enc.return_value = mock_encoder

        mock_collection = MagicMock()
        hit = _make_mock_hit(0.8, "docs/b.md", "https://b", "content")
        mock_collection.search.return_value = [[hit]]
        mock_coll_cls.return_value = mock_collection

        _search("test query", partition="")

        call_kwargs = mock_collection.search.call_args[1]
        self.assertNotIn("partition_names", call_kwargs)

    @patch("agent.mcp_server._get_encoder")
    @patch("agent.mcp_server._ensure_milvus")
    @patch("agent.mcp_server.Collection")
    def test_search_broadens_on_low_similarity(self, mock_coll_cls, mock_ensure, mock_enc):
        """When avg similarity < threshold, search should retry with broadened query."""
        from agent.mcp_server import _search

        mock_encoder = MagicMock()
        call_count = [0]

        def encode_side_effect(text):
            call_count[0] += 1
            mock_vec = MagicMock()
            mock_vec.tolist.return_value = [0.1] * 768
            return mock_vec

        mock_encoder.encode = encode_side_effect
        mock_enc.return_value = mock_encoder

        mock_collection = MagicMock()
        # First call: low similarity
        low_hit = _make_mock_hit(0.1, "a.md", "https://a", "low")
        # Second call: high similarity
        high_hit = _make_mock_hit(0.9, "b.md", "https://b", "high")
        mock_collection.search.side_effect = [[[low_hit]], [[high_hit]]]
        mock_coll_cls.return_value = mock_collection

        hits = _search("test", auto_broaden=True)

        # Should have made 2 encode calls (original + broadened)
        self.assertEqual(call_count[0], 2)
        # Should return the high-quality results
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0]["similarity"], 0.9, places=2)

    @patch("agent.mcp_server._get_encoder")
    @patch("agent.mcp_server._ensure_milvus")
    @patch("agent.mcp_server.Collection")
    def test_search_returns_best_after_max_attempts(self, mock_coll_cls, mock_ensure, mock_enc):
        """When all attempts yield low similarity, return the best seen."""
        from agent.mcp_server import _search

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 768)
        mock_enc.return_value = mock_encoder

        mock_collection = MagicMock()
        low_hit = _make_mock_hit(0.15, "a.md", "https://a", "low quality")
        mock_collection.search.return_value = [[low_hit]]
        mock_coll_cls.return_value = mock_collection

        hits = _search("obscure query", auto_broaden=True)

        # Should return the low-quality results rather than nothing
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0]["similarity"], 0.15, places=2)

    @patch("agent.mcp_server._get_encoder")
    @patch("agent.mcp_server._ensure_milvus")
    @patch("agent.mcp_server.Collection")
    def test_search_empty_results(self, mock_coll_cls, mock_ensure, mock_enc):
        """No hits from Milvus should return an empty list."""
        from agent.mcp_server import _search

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 768)
        mock_enc.return_value = mock_encoder

        mock_collection = MagicMock()
        mock_collection.search.return_value = [[]]
        mock_coll_cls.return_value = mock_collection

        hits = _search("nothing matches")
        self.assertEqual(hits, [])

    @patch("agent.mcp_server._get_encoder")
    @patch("agent.mcp_server._ensure_milvus")
    @patch("agent.mcp_server.Collection")
    def test_search_truncates_long_content(self, mock_coll_cls, mock_ensure, mock_enc):
        """Content longer than 500 chars should be truncated."""
        from agent.mcp_server import _search

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 768)
        mock_enc.return_value = mock_encoder

        mock_collection = MagicMock()
        long_text = "x" * 1000
        hit = _make_mock_hit(0.8, "a.md", "https://a", long_text)
        mock_collection.search.return_value = [[hit]]
        mock_coll_cls.return_value = mock_collection

        hits = _search("test", auto_broaden=False)
        self.assertTrue(hits[0]["content_text"].endswith("..."))
        self.assertLessEqual(len(hits[0]["content_text"]), 504)  # 500 + "..."


class TestMCPTools(unittest.TestCase):
    """Test the MCP tool functions (search_kubeflow_docs, etc.)."""

    @patch("agent.mcp_server._search")
    def test_search_kubeflow_docs_calls_search_with_correct_partition(self, mock_search):
        from agent.mcp_server import search_kubeflow_docs
        from agent.config import PARTITION_MAP, Intent

        mock_search.return_value = []
        search_kubeflow_docs("test query", top_k=3)

        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        self.assertEqual(call_kwargs["top_k"], 3)
        self.assertEqual(
            call_kwargs["partition"],
            PARTITION_MAP.get(Intent.KUBEFLOW_DOCS, ""),
        )

    @patch("agent.mcp_server._search")
    def test_search_platform_docs_uses_platform_partition(self, mock_search):
        from agent.mcp_server import search_platform_docs
        from agent.config import PARTITION_MAP, Intent

        mock_search.return_value = []
        search_platform_docs("terraform oke", top_k=5)

        call_kwargs = mock_search.call_args[1]
        self.assertEqual(
            call_kwargs["partition"],
            PARTITION_MAP.get(Intent.PLATFORM_ARCH, "platform_arch"),
        )

    @patch("agent.mcp_server._search")
    def test_search_all_docs_uses_empty_partition(self, mock_search):
        from agent.mcp_server import search_all_docs

        mock_search.return_value = []
        search_all_docs("cross-cutting query")

        call_kwargs = mock_search.call_args[1]
        self.assertEqual(call_kwargs["partition"], "")

    @patch("agent.mcp_server._search")
    def test_top_k_clamped(self, mock_search):
        """top_k outside 1-10 should be clamped."""
        from agent.mcp_server import search_kubeflow_docs

        mock_search.return_value = []

        search_kubeflow_docs("test", top_k=0)
        self.assertEqual(mock_search.call_args[1]["top_k"], 1)

        search_kubeflow_docs("test", top_k=99)
        self.assertEqual(mock_search.call_args[1]["top_k"], 10)

    @patch("agent.mcp_server._search")
    def test_tool_returns_error_message_on_exception(self, mock_search):
        from agent.mcp_server import search_kubeflow_docs

        mock_search.side_effect = RuntimeError("Milvus down")
        result = search_kubeflow_docs("test")
        self.assertIn("Search failed", result)
        self.assertIn("internal error", result)


class TestEncoderSingleton(unittest.TestCase):
    """Test thread-safe encoder initialisation."""

    @patch("agent.mcp_server.SentenceTransformer")
    def test_encoder_loaded_once(self, mock_st_cls):
        """Multiple calls to _get_encoder should only create one instance."""
        import agent.mcp_server as mod

        # Reset singleton state
        mod._encoder = None
        mock_st_cls.return_value = MagicMock()

        results = []

        def load():
            results.append(mod._get_encoder())

        threads = [threading.Thread(target=load) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        self.assertTrue(all(r is results[0] for r in results))
        # SentenceTransformer should have been instantiated exactly once
        self.assertEqual(mock_st_cls.call_count, 1)

        # Cleanup
        mod._encoder = None


class TestMilvusConnection(unittest.TestCase):
    """Test _ensure_milvus() reconnection logic."""

    @patch("agent.mcp_server.connections")
    @patch("agent.mcp_server.utility")
    def test_reconnects_on_stale_connection(self, mock_utility, mock_conns):
        import agent.mcp_server as mod

        mod._milvus_connected = True
        mock_utility.list_collections.side_effect = Exception("stale")

        mod._ensure_milvus()

        mock_conns.connect.assert_called_once()
        self.assertTrue(mod._milvus_connected)

        # Cleanup
        mod._milvus_connected = False

    @patch("agent.mcp_server.connections")
    @patch("agent.mcp_server.utility")
    def test_no_reconnect_when_healthy(self, mock_utility, mock_conns):
        import agent.mcp_server as mod

        mod._milvus_connected = True
        mock_utility.list_collections.return_value = ["test"]

        mod._ensure_milvus()

        mock_conns.connect.assert_not_called()

        # Cleanup
        mod._milvus_connected = False


if __name__ == "__main__":
    unittest.main()
