"""Tests for reranker_client and compressor."""

from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "docs-agent-mcp" / "mcp-server"))

import reranker_client
import compressor


def test_reranker_client_reranks_hits():
    hits = [
        {"entity": {"content_text": "doc1"}},
        {"entity": {"content_text": "doc2"}},
    ]
    with patch("reranker_client.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"index": 1, "score": 0.9},
            {"index": 0, "score": 0.1},
        ]
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"RERANKER_URL": "http://mock"}):
            reranked = reranker_client.rerank_hits("query", hits)

        assert len(reranked) == 2
        assert reranked[0]["entity"]["content_text"] == "doc2"
        assert reranked[1]["entity"]["content_text"] == "doc1"
        assert reranked[0]["distance"] == 0.9
        assert reranked[1]["distance"] == 0.1
        assert reranked[0]["reranked"] is True


def test_reranker_client_bypasses_if_no_url():
    hits = [{"entity": {"content_text": "doc1"}}]
    with patch.dict("os.environ", {"RERANKER_URL": ""}):
        reranked = reranker_client.rerank_hits("query", hits)
        assert reranked == hits


def test_compressor_compresses_hits():
    hits = [
        {"entity": {"content_text": "First sentence. Second sentence. Third sentence."}},
    ]
    with patch("compressor.embeddings_client.embed_query") as mock_embed_query:
        with patch("compressor.embeddings_client.embed_texts") as mock_embed_texts:
            mock_embed_query.return_value = [1.0, 0.0]
            # First and third sentences match the query perfectly
            mock_embed_texts.return_value = [
                [1.0, 0.0],  # First sentence
                [0.0, 1.0],  # Second sentence
                [1.0, 0.0],  # Third sentence
            ]

            # Allow only enough chars for the first and third sentences (approx 35 chars)
            max_chars = 35
            compressed = compressor.compress_hits("query", hits, max_chars)

            assert len(compressed) == 1
            # Note: "First sentence." + "Third sentence." = 15 + 15 = 30 chars
            assert "First sentence." in compressed[0]["entity"]["content_text"]
            assert "Third sentence." in compressed[0]["entity"]["content_text"]
            assert "Second sentence." not in compressed[0]["entity"]["content_text"]


def test_compressor_bypasses_if_no_hits():
    compressed = compressor.compress_hits("query", [], 100)
    assert compressed == []


def test_compressor_bypasses_if_max_chars_zero():
    hits = [{"entity": {"content_text": "doc1"}}]
    compressed = compressor.compress_hits("query", hits, 0)
    assert compressed == hits
